"""
Collaborative Features Module for Manus Clone

This module implements collaborative features including:
- User management and authentication
- Project sharing and permissions
- Real-time collaboration
- Notifications and alerts
"""

import os
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CollaborationManager:
    """Manages collaborative features for the Manus Clone application"""
    
    def __init__(self, app: Flask, config: Dict[str, Any]):
        """
        Initialize the Collaboration Manager with configuration
        
        Args:
            app: Flask application instance
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.app = app
        
        # Initialize JWT for authentication
        self.jwt = JWTManager(app)
        app.config['JWT_SECRET_KEY'] = config.get('jwt_secret_key', os.urandom(24).hex())
        app.config['JWT_ACCESS_TOKEN_EXPIRES'] = config.get('jwt_token_expires', 86400)  # 24 hours by default
        
        # Initialize SocketIO for real-time collaboration
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        
        # User data storage
        self.users_dir = os.path.join(config.get('data_dir', 'data'), 'users')
        os.makedirs(self.users_dir, exist_ok=True)
        
        # Project sharing data storage
        self.shares_dir = os.path.join(config.get('data_dir', 'data'), 'shares')
        os.makedirs(self.shares_dir, exist_ok=True)
        
        # Notification data storage
        self.notifications_dir = os.path.join(config.get('data_dir', 'data'), 'notifications')
        os.makedirs(self.notifications_dir, exist_ok=True)
        
        # Active users tracking
        self.active_users = {}
        
        # Register routes
        self._register_routes()
        
        # Register socket events
        self._register_socket_events()
        
        logger.info("Collaboration Manager initialized successfully")
    
    def _register_routes(self):
        """Register HTTP routes for collaboration features"""
        
        @self.app.route('/api/auth/register', methods=['POST'])
        def register_user():
            """Register a new user"""
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['username', 'email', 'password', 'full_name']
            for field in required_fields:
                if field not in data:
                    return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
            
            # Check if username or email already exists
            if self._user_exists(data['username'], data['email']):
                return jsonify({'success': False, 'error': 'Username or email already exists'}), 409
            
            # Create user object
            user_id = str(uuid.uuid4())
            user = {
                'id': user_id,
                'username': data['username'],
                'email': data['email'],
                'full_name': data['full_name'],
                'password_hash': self._hash_password(data['password']),
                'role': data.get('role', 'user'),
                'created_at': int(time.time()),
                'last_login': None,
                'settings': data.get('settings', {}),
                'profile_image': data.get('profile_image', None)
            }
            
            # Save user data
            self._save_user(user)
            
            # Create access token
            access_token = create_access_token(identity=user_id)
            
            return jsonify({
                'success': True,
                'message': 'User registered successfully',
                'user_id': user_id,
                'access_token': access_token
            })
        
        @self.app.route('/api/auth/login', methods=['POST'])
        def login_user():
            """Login a user"""
            data = request.get_json()
            
            # Validate required fields
            if 'username' not in data or 'password' not in data:
                return jsonify({'success': False, 'error': 'Missing username or password'}), 400
            
            # Get user by username
            user = self._get_user_by_username(data['username'])
            if not user:
                return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
            
            # Verify password
            if not self._verify_password(data['password'], user['password_hash']):
                return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
            
            # Update last login time
            user['last_login'] = int(time.time())
            self._save_user(user)
            
            # Create access token
            access_token = create_access_token(identity=user['id'])
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user_id': user['id'],
                'access_token': access_token,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'full_name': user['full_name'],
                    'role': user['role'],
                    'profile_image': user['profile_image']
                }
            })
        
        @self.app.route('/api/users/profile', methods=['GET'])
        @jwt_required()
        def get_user_profile():
            """Get the current user's profile"""
            user_id = get_jwt_identity()
            user = self._get_user_by_id(user_id)
            
            if not user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'full_name': user['full_name'],
                    'role': user['role'],
                    'created_at': user['created_at'],
                    'last_login': user['last_login'],
                    'profile_image': user['profile_image'],
                    'settings': user['settings']
                }
            })
        
        @self.app.route('/api/users/profile', methods=['PUT'])
        @jwt_required()
        def update_user_profile():
            """Update the current user's profile"""
            user_id = get_jwt_identity()
            user = self._get_user_by_id(user_id)
            
            if not user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
            
            data = request.get_json()
            
            # Update user fields
            updatable_fields = ['full_name', 'email', 'profile_image', 'settings']
            for field in updatable_fields:
                if field in data:
                    user[field] = data[field]
            
            # Update password if provided
            if 'password' in data and data['password']:
                user['password_hash'] = self._hash_password(data['password'])
            
            # Save updated user data
            self._save_user(user)
            
            return jsonify({
                'success': True,
                'message': 'Profile updated successfully',
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'full_name': user['full_name'],
                    'role': user['role'],
                    'profile_image': user['profile_image'],
                    'settings': user['settings']
                }
            })
        
        @self.app.route('/api/projects/<project_id>/share', methods=['POST'])
        @jwt_required()
        def share_project(project_id):
            """Share a project with another user"""
            user_id = get_jwt_identity()
            data = request.get_json()
            
            # Validate required fields
            if 'username' not in data or 'permission' not in data:
                return jsonify({'success': False, 'error': 'Missing username or permission'}), 400
            
            # Validate permission level
            valid_permissions = ['view', 'edit', 'admin']
            if data['permission'] not in valid_permissions:
                return jsonify({'success': False, 'error': f'Invalid permission. Must be one of: {", ".join(valid_permissions)}'}), 400
            
            # Get target user
            target_user = self._get_user_by_username(data['username'])
            if not target_user:
                return jsonify({'success': False, 'error': 'User not found'}), 404
            
            # Create share record
            share_id = str(uuid.uuid4())
            share = {
                'id': share_id,
                'project_id': project_id,
                'owner_id': user_id,
                'user_id': target_user['id'],
                'permission': data['permission'],
                'created_at': int(time.time()),
                'last_accessed': None
            }
            
            # Save share data
            self._save_share(share)
            
            # Create notification for target user
            owner = self._get_user_by_id(user_id)
            self._create_notification(
                target_user['id'],
                'project_shared',
                f"{owner['full_name']} shared a project with you",
                {'project_id': project_id, 'permission': data['permission']}
            )
            
            return jsonify({
                'success': True,
                'message': 'Project shared successfully',
                'share_id': share_id
            })
        
        @self.app.route('/api/projects/shared', methods=['GET'])
        @jwt_required()
        def get_shared_projects():
            """Get projects shared with the current user"""
            user_id = get_jwt_identity()
            
            # Get all shares for this user
            shares = self._get_shares_for_user(user_id)
            
            return jsonify({
                'success': True,
                'shares': shares
            })
        
        @self.app.route('/api/notifications', methods=['GET'])
        @jwt_required()
        def get_notifications():
            """Get notifications for the current user"""
            user_id = get_jwt_identity()
            
            # Get all notifications for this user
            notifications = self._get_notifications_for_user(user_id)
            
            return jsonify({
                'success': True,
                'notifications': notifications
            })
        
        @self.app.route('/api/notifications/<notification_id>/read', methods=['POST'])
        @jwt_required()
        def mark_notification_read(notification_id):
            """Mark a notification as read"""
            user_id = get_jwt_identity()
            
            # Get the notification
            notification = self._get_notification(notification_id)
            
            if not notification:
                return jsonify({'success': False, 'error': 'Notification not found'}), 404
            
            # Verify the notification belongs to this user
            if notification['user_id'] != user_id:
                return jsonify({'success': False, 'error': 'Unauthorized'}), 403
            
            # Mark as read
            notification['read'] = True
            notification['read_at'] = int(time.time())
            
            # Save updated notification
            self._save_notification(notification)
            
            return jsonify({
                'success': True,
                'message': 'Notification marked as read'
            })
    
    def _register_socket_events(self):
        """Register SocketIO events for real-time collaboration"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
            
            # Remove user from active users and rooms
            for user_id, data in list(self.active_users.items()):
                if request.sid in data['sessions']:
                    data['sessions'].remove(request.sid)
                    if not data['sessions']:
                        del self.active_users[user_id]
                    break
        
        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """Authenticate user for socket connection"""
            token = data.get('token')
            if not token:
                emit('error', {'message': 'Authentication required'})
                return
            
            try:
                # Verify token and get user ID
                user_id = self.jwt._decode_jwt_from_config(token).get('sub')
                user = self._get_user_by_id(user_id)
                
                if not user:
                    emit('error', {'message': 'User not found'})
                    return
                
                # Add user to active users
                if user_id not in self.active_users:
                    self.active_users[user_id] = {
                        'user': user,
                        'sessions': [request.sid]
                    }
                else:
                    self.active_users[user_id]['sessions'].append(request.sid)
                
                emit('authenticated', {
                    'user_id': user_id,
                    'username': user['username']
                })
                
                logger.info(f"User authenticated: {user['username']} ({user_id})")
                
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                emit('error', {'message': 'Authentication failed'})
        
        @self.socketio.on('join_project')
        def handle_join_project(data):
            """Join a project room for real-time collaboration"""
            project_id = data.get('project_id')
            if not project_id:
                emit('error', {'message': 'Project ID required'})
                return
            
            # Find user ID from session
            user_id = None
            for uid, data in self.active_users.items():
                if request.sid in data['sessions']:
                    user_id = uid
                    break
            
            if not user_id:
                emit('error', {'message': 'Not authenticated'})
                return
            
            # Check if user has access to this project
            if not self._user_has_project_access(user_id, project_id):
                emit('error', {'message': 'Access denied'})
                return
            
            # Join the project room
            room_name = f"project_{project_id}"
            join_room(room_name)
            
            # Notify others in the room
            user = self._get_user_by_id(user_id)
            emit('user_joined', {
                'user_id': user_id,
                'username': user['username'],
                'full_name': user['full_name']
            }, room=room_name, include_self=False)
            
            logger.info(f"User {user['username']} joined project {project_id}")
            
            # Send current active users in this project
            active_project_users = []
            for uid, data in self.active_users.items():
                if uid != user_id and self._user_has_project_access(uid, project_id):
                    active_project_users.append({
                        'user_id': uid,
                        'username': data['user']['username'],
                        'full_name': data['user']['full_name']
                    })
            
            emit('active_users', {'users': active_project_users})
        
        @self.socketio.on('leave_project')
        def handle_leave_project(data):
            """Leave a project room"""
            project_id = data.get('project_id')
            if not project_id:
                emit('error', {'message': 'Project ID required'})
                return
            
            # Find user ID from session
            user_id = None
            for uid, data in self.active_users.items():
                if request.sid in data['sessions']:
                    user_id = uid
                    break
            
            if not user_id:
                emit('error', {'message': 'Not authenticated'})
                return
            
            # Leave the project room
            room_name = f"project_{project_id}"
            leave_room(room_name)
            
            # Notify others in the room
            user = self._get_user_by_id(user_id)
            emit('user_left', {
                'user_id': user_id,
                'username': user['username']
            }, room=room_name)
            
            logger.info(f"User {user['username']} left project {project_id}")
        
        @self.socketio.on('project_update')
        def handle_project_update(data):
            """Handle project updates and broadcast to collaborators"""
            project_id = data.get('project_id')
            update_type = data.get('type')
            update_data = data.get('data', {})
            
            if not project_id or not update_type:
                emit('error', {'message': 'Project ID and update type required'})
                return
            
            # Find user ID from session
            user_id = None
            for uid, data in self.active_users.items():
                if request.sid in data['sessions']:
                    user_id = uid
                    break
            
            if not user_id:
                emit('error', {'message': 'Not authenticated'})
                return
            
            # Check if user has edit access to this project
            if not self._user_has_project_access(user_id, project_id, min_permission='edit'):
                emit('error', {'message': 'Edit access required'})
                return
            
            # Get user info
            user = self._get_user_by_id(user_id)
            
            # Broadcast the update to all users in the project room
            room_name = f"project_{project_id}"
            emit('project_updated', {
                'type': update_type,
                'data': update_data,
                'user': {
                    'id': user_id,
                    'username': user['username'],
                    'full_name': user['full_name']
                },
                'timestamp': int(time.time())
            }, room=room_name, include_self=False)
            
            logger.info(f"Project update: {project_id}, type: {update_type}, by: {user['username']}")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password for secure storage"""
        # In a production environment, use a proper password hashing library like bcrypt
        # This is a simplified implementation for demonstration purposes
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        # In a production environment, use a proper password hashing library like bcrypt
        # This is a simplified implementation for demonstration purposes
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest() == password_hash
    
    def _user_exists(self, username: str, email: str) -> bool:
        """Check if a user with the given username or email exists"""
        for filename in os.listdir(self.users_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.users_dir, filename), 'r') as f:
                    user = json.load(f)
                    if user['username'] == username or user['email'] == email:
                        return True
        return False
    
    def _save_user(self, user: Dict[str, Any]) -> None:
        """Save user data to file"""
        user_path = os.path.join(self.users_dir, f"{user['id']}.json")
        with open(user_path, 'w') as f:
            json.dump(user, f, indent=2)
    
    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID"""
        user_path = os.path.join(self.users_dir, f"{user_id}.json")
        if os.path.exists(user_path):
            with open(user_path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get a user by username"""
        for filename in os.listdir(self.users_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.users_dir, filename), 'r') as f:
                    user = json.load(f)
                    if user['username'] == username:
                        return user
        return None
    
    def _save_share(self, share: Dict[str, Any]) -> None:
        """Save share data to file"""
        share_path = os.path.join(self.shares_dir, f"{share['id']}.json")
        with open(share_path, 'w') as f:
            json.dump(share, f, indent=2)
    
    def _get_shares_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all shares for a user"""
        shares = []
        for filename in os.listdir(self.shares_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.shares_dir, filename), 'r') as f:
                    share = json.load(f)
                    if share['user_id'] == user_id:
                        shares.append(share)
        return shares
    
    def _user_has_project_access(self, user_id: str, project_id: str, min_permission: str = 'view') -> bool:
        """Check if a user has access to a project with at least the specified permission level"""
        # Permission hierarchy: view < edit < admin
        permission_levels = {
            'view': 0,
            'edit': 1,
            'admin': 2
        }
        min_level = permission_levels.get(min_permission, 0)
        
        # Check if user is the project owner (has all permissions)
        # In a real implementation, you would check the project's owner field
        # This is a simplified implementation
        
        # Check if project is shared with the user
        for filename in os.listdir(self.shares_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.shares_dir, filename), 'r') as f:
                    share = json.load(f)
                    if share['project_id'] == project_id and share['user_id'] == user_id:
                        # Check if the user's permission level is sufficient
                        user_level = permission_levels.get(share['permission'], 0)
                        if user_level >= min_level:
                            return True
        
        return False
    
    def _create_notification(self, user_id: str, notification_type: str, message: str, data: Dict[str, Any] = None) -> str:
        """Create a notification for a user"""
        notification_id = str(uuid.uuid4())
        notification = {
            'id': notification_id,
            'user_id': user_id,
            'type': notification_type,
            'message': message,
            'data': data or {},
            'created_at': int(time.time()),
            'read': False,
            'read_at': None
        }
        
        # Save notification
        self._save_notification(notification)
        
        # Send real-time notification if user is online
        if user_id in self.active_users:
            for session_id in self.active_users[user_id]['sessions']:
                self.socketio.emit('notification', notification, room=session_id)
        
        return notification_id
    
    def _save_notification(self, notification: Dict[str, Any]) -> None:
        """Save notification data to file"""
        notification_path = os.path.join(self.notifications_dir, f"{notification['id']}.json")
        with open(notification_path, 'w') as f:
            json.dump(notification, f, indent=2)
    
    def _get_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get a notification by ID"""
        notification_path = os.path.join(self.notifications_dir, f"{notification_id}.json")
        if os.path.exists(notification_path):
            with open(notification_path, 'r') as f:
                return json.load(f)
        return None
    
    def _get_notifications_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all notifications for a user"""
        notifications = []
        for filename in os.listdir(self.notifications_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.notifications_dir, filename), 'r') as f:
                    notification = json.load(f)
                    if notification['user_id'] == user_id:
                        notifications.append(notification)
        
        # Sort by creation time (newest first)
        notifications.sort(key=lambda x: x['created_at'], reverse=True)
        
        return notifications

# Function to initialize and register the collaboration features with the Flask app
def init_collaborative_features(app: Flask, config: Dict[str, Any]) -> CollaborationManager:
    """Initialize and register collaborative features with the Flask app"""
    collaboration_manager = CollaborationManager(app, config)
    return collaboration_manager
