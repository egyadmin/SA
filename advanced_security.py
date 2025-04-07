"""
Advanced Security and Privacy Module for Manus Clone

This module implements advanced security features including:
- Multi-factor authentication
- Data encryption
- Privacy policies and permissions management
- Security event monitoring and logging
"""

import os
import json
import time
import logging
import uuid
import base64
import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session, abort
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity
import pyotp
import qrcode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages advanced security features for the Manus Clone application"""
    
    def __init__(self, app: Flask, config: Dict[str, Any]):
        """
        Initialize the Security Manager with configuration
        
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
        app.config['JWT_REFRESH_TOKEN_EXPIRES'] = config.get('jwt_refresh_token_expires', 2592000)  # 30 days by default
        
        # Security data storage
        self.security_dir = os.path.join(config.get('data_dir', 'data'), 'security')
        os.makedirs(self.security_dir, exist_ok=True)
        
        # MFA data storage
        self.mfa_dir = os.path.join(self.security_dir, 'mfa')
        os.makedirs(self.mfa_dir, exist_ok=True)
        
        # Encryption keys storage
        self.keys_dir = os.path.join(self.security_dir, 'keys')
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Security logs storage
        self.logs_dir = os.path.join(self.security_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Privacy policies storage
        self.policies_dir = os.path.join(self.security_dir, 'policies')
        os.makedirs(self.policies_dir, exist_ok=True)
        
        # Initialize master encryption key
        self._init_master_key()
        
        # Register routes
        self._register_routes()
        
        # Register JWT callbacks
        self._register_jwt_callbacks()
        
        logger.info("Security Manager initialized successfully")
    
    def _init_master_key(self):
        """Initialize or load the master encryption key"""
        master_key_path = os.path.join(self.keys_dir, 'master.key')
        
        if os.path.exists(master_key_path):
            # Load existing master key
            with open(master_key_path, 'rb') as f:
                self.master_key = f.read()
        else:
            # Generate new master key
            self.master_key = Fernet.generate_key()
            
            # Save master key
            with open(master_key_path, 'wb') as f:
                f.write(self.master_key)
        
        # Initialize Fernet cipher with master key
        self.cipher = Fernet(self.master_key)
    
    def _register_routes(self):
        """Register HTTP routes for security features"""
        
        @self.app.route('/api/auth/mfa/setup', methods=['POST'])
        @jwt_required()
        def setup_mfa():
            """Set up multi-factor authentication for a user"""
            user_id = get_jwt_identity()
            
            # Generate a new TOTP secret
            totp_secret = pyotp.random_base32()
            
            # Create MFA record
            mfa_data = {
                'user_id': user_id,
                'secret': totp_secret,
                'enabled': False,
                'created_at': int(time.time()),
                'verified_at': None,
                'backup_codes': self._generate_backup_codes()
            }
            
            # Save MFA data
            self._save_mfa_data(user_id, mfa_data)
            
            # Generate QR code for TOTP app
            app_name = self.config.get('app_name', 'ManusClone')
            user_email = self._get_user_email(user_id)
            totp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
                name=user_email,
                issuer_name=app_name
            )
            
            # Generate QR code image
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_path = os.path.join(self.mfa_dir, f"{user_id}_qr.png")
            qr_img.save(qr_path)
            
            return jsonify({
                'success': True,
                'message': 'MFA setup initiated',
                'secret': totp_secret,
                'qr_code_url': f"/api/auth/mfa/qrcode/{user_id}",
                'backup_codes': mfa_data['backup_codes']
            })
        
        @self.app.route('/api/auth/mfa/qrcode/<user_id>', methods=['GET'])
        @jwt_required()
        def get_mfa_qrcode(user_id):
            """Get the QR code for MFA setup"""
            current_user_id = get_jwt_identity()
            
            # Verify the user is requesting their own QR code
            if current_user_id != user_id:
                return jsonify({'success': False, 'error': 'Unauthorized'}), 403
            
            qr_path = os.path.join(self.mfa_dir, f"{user_id}_qr.png")
            
            if not os.path.exists(qr_path):
                return jsonify({'success': False, 'error': 'QR code not found'}), 404
            
            # In a real implementation, you would return the image file
            # For this example, we'll just return a success message
            return jsonify({
                'success': True,
                'message': 'QR code available',
                'qr_code_path': qr_path
            })
        
        @self.app.route('/api/auth/mfa/verify', methods=['POST'])
        @jwt_required()
        def verify_mfa():
            """Verify MFA code and enable MFA for a user"""
            user_id = get_jwt_identity()
            data = request.get_json()
            
            if 'code' not in data:
                return jsonify({'success': False, 'error': 'MFA code required'}), 400
            
            # Get MFA data
            mfa_data = self._get_mfa_data(user_id)
            
            if not mfa_data:
                return jsonify({'success': False, 'error': 'MFA not set up'}), 404
            
            # Verify the code
            totp = pyotp.TOTP(mfa_data['secret'])
            if totp.verify(data['code']):
                # Update MFA data
                mfa_data['enabled'] = True
                mfa_data['verified_at'] = int(time.time())
                self._save_mfa_data(user_id, mfa_data)
                
                # Log security event
                self._log_security_event(user_id, 'mfa_enabled', 'MFA enabled successfully')
                
                return jsonify({
                    'success': True,
                    'message': 'MFA verified and enabled successfully'
                })
            else:
                # Log failed verification attempt
                self._log_security_event(user_id, 'mfa_verification_failed', 'Failed MFA verification attempt')
                
                return jsonify({
                    'success': False,
                    'error': 'Invalid MFA code'
                }), 400
        
        @self.app.route('/api/auth/mfa/disable', methods=['POST'])
        @jwt_required()
        def disable_mfa():
            """Disable MFA for a user"""
            user_id = get_jwt_identity()
            data = request.get_json()
            
            if 'password' not in data:
                return jsonify({'success': False, 'error': 'Password required'}), 400
            
            # Verify password (in a real implementation, you would check against the user's stored password)
            if not self._verify_user_password(user_id, data['password']):
                # Log failed password verification
                self._log_security_event(user_id, 'mfa_disable_failed', 'Failed password verification for MFA disable')
                
                return jsonify({'success': False, 'error': 'Invalid password'}), 400
            
            # Get MFA data
            mfa_data = self._get_mfa_data(user_id)
            
            if not mfa_data:
                return jsonify({'success': False, 'error': 'MFA not set up'}), 404
            
            # Disable MFA
            mfa_data['enabled'] = False
            self._save_mfa_data(user_id, mfa_data)
            
            # Log security event
            self._log_security_event(user_id, 'mfa_disabled', 'MFA disabled')
            
            return jsonify({
                'success': True,
                'message': 'MFA disabled successfully'
            })
        
        @self.app.route('/api/auth/login/mfa', methods=['POST'])
        def login_mfa():
            """Second step of login with MFA"""
            data = request.get_json()
            
            if 'user_id' not in data or 'code' not in data:
                return jsonify({'success': False, 'error': 'User ID and MFA code required'}), 400
            
            user_id = data['user_id']
            mfa_code = data['code']
            
            # Get MFA data
            mfa_data = self._get_mfa_data(user_id)
            
            if not mfa_data or not mfa_data.get('enabled', False):
                return jsonify({'success': False, 'error': 'MFA not enabled for this user'}), 400
            
            # Check if using a backup code
            if mfa_code in mfa_data.get('backup_codes', []):
                # Remove the used backup code
                mfa_data['backup_codes'].remove(mfa_code)
                self._save_mfa_data(user_id, mfa_data)
                
                # Log backup code usage
                self._log_security_event(user_id, 'mfa_backup_code_used', 'Backup code used for MFA')
                
                # Generate tokens
                access_token = create_access_token(identity=user_id)
                refresh_token = create_refresh_token(identity=user_id)
                
                return jsonify({
                    'success': True,
                    'message': 'MFA verified with backup code',
                    'access_token': access_token,
                    'refresh_token': refresh_token
                })
            
            # Verify TOTP code
            totp = pyotp.TOTP(mfa_data['secret'])
            if totp.verify(mfa_code):
                # Log successful MFA
                self._log_security_event(user_id, 'mfa_login_success', 'Successful MFA login')
                
                # Generate tokens
                access_token = create_access_token(identity=user_id)
                refresh_token = create_refresh_token(identity=user_id)
                
                return jsonify({
                    'success': True,
                    'message': 'MFA verified successfully',
                    'access_token': access_token,
                    'refresh_token': refresh_token
                })
            else:
                # Log failed MFA attempt
                self._log_security_event(user_id, 'mfa_login_failed', 'Failed MFA login attempt')
                
                return jsonify({
                    'success': False,
                    'error': 'Invalid MFA code'
                }), 400
        
        @self.app.route('/api/auth/refresh', methods=['POST'])
        @jwt_required(refresh=True)
        def refresh_token():
            """Refresh access token"""
            user_id = get_jwt_identity()
            
            # Generate new access token
            access_token = create_access_token(identity=user_id)
            
            # Log token refresh
            self._log_security_event(user_id, 'token_refresh', 'Access token refreshed')
            
            return jsonify({
                'success': True,
                'access_token': access_token
            })
        
        @self.app.route('/api/security/encrypt', methods=['POST'])
        @jwt_required()
        def encrypt_data():
            """Encrypt sensitive data"""
            user_id = get_jwt_identity()
            data = request.get_json()
            
            if 'data' not in data:
                return jsonify({'success': False, 'error': 'Data required for encryption'}), 400
            
            # Encrypt the data
            encrypted_data = self.encrypt_data(data['data'])
            
            # Log encryption event
            self._log_security_event(user_id, 'data_encrypted', 'Data encrypted')
            
            return jsonify({
                'success': True,
                'encrypted_data': encrypted_data
            })
        
        @self.app.route('/api/security/decrypt', methods=['POST'])
        @jwt_required()
        def decrypt_data():
            """Decrypt encrypted data"""
            user_id = get_jwt_identity()
            data = request.get_json()
            
            if 'encrypted_data' not in data:
                return jsonify({'success': False, 'error': 'Encrypted data required for decryption'}), 400
            
            try:
                # Decrypt the data
                decrypted_data = self.decrypt_data(data['encrypted_data'])
                
                # Log decryption event
                self._log_security_event(user_id, 'data_decrypted', 'Data decrypted')
                
                return jsonify({
                    'success': True,
                    'decrypted_data': decrypted_data
                })
            except Exception as e:
                # Log decryption failure
                self._log_security_event(user_id, 'decryption_failed', f'Decryption failed: {str(e)}')
                
                return jsonify({
                    'success': False,
                    'error': 'Decryption failed'
                }), 400
        
        @self.app.route('/api/security/logs', methods=['GET'])
        @jwt_required()
        def get_security_logs():
            """Get security logs for a user"""
            user_id = get_jwt_identity()
            
            # Check if user has admin role (in a real implementation, you would check user roles)
            is_admin = self._is_admin_user(user_id)
            
            # Get query parameters
            target_user_id = request.args.get('user_id', user_id)
            event_type = request.args.get('event_type')
            start_time = request.args.get('start_time')
            end_time = request.args.get('end_time')
            limit = int(request.args.get('limit', 100))
            
            # Non-admin users can only view their own logs
            if target_user_id != user_id and not is_admin:
                return jsonify({'success': False, 'error': 'Unauthorized'}), 403
            
            # Get logs
            logs = self._get_security_logs(
                user_id=target_user_id,
                event_type=event_type,
                start_time=int(start_time) if start_time else None,
                end_time=int(end_time) if end_time else None,
                limit=limit
            )
            
            return jsonify({
                'success': True,
                'logs': logs
            })
        
        @self.app.route('/api/security/policy', methods=['GET'])
        def get_privacy_policy():
            """Get the current privacy policy"""
            # Get the latest privacy policy
            policy = self._get_latest_privacy_policy()
            
            if not policy:
                return jsonify({'success': False, 'error': 'Privacy policy not found'}), 404
            
            return jsonify({
                'success': True,
                'policy': policy
            })
        
        @self.app.route('/api/security/policy', methods=['POST'])
        @jwt_required()
        def update_privacy_policy():
            """Update the privacy policy (admin only)"""
            user_id = get_jwt_identity()
            
            # Check if user has admin role
            if not self._is_admin_user(user_id):
                return jsonify({'success': False, 'error': 'Unauthorized'}), 403
            
            data = request.get_json()
            
            if 'content' not in data:
                return jsonify({'success': False, 'error': 'Policy content required'}), 400
            
            # Create new policy version
            policy_id = str(uuid.uuid4())
            policy = {
                'id': policy_id,
                'version': self._get_next_policy_version(),
                'content': data['content'],
                'created_at': int(time.time()),
                'created_by': user_id
            }
            
            # Save policy
            self._save_privacy_policy(policy)
            
            # Log policy update
            self._log_security_event(user_id, 'policy_updated', f'Privacy policy updated to version {policy["version"]}')
            
            return jsonify({
                'success': True,
                'message': 'Privacy policy updated successfully',
                'policy_id': policy_id,
                'version': policy['version']
            })
        
        @self.app.route('/api/security/policy/consent', methods=['POST'])
        @jwt_required()
        def consent_to_policy():
            """Record user consent to privacy policy"""
            user_id = get_jwt_identity()
            data = request.get_json()
            
            if 'version' not in data:
                return jsonify({'success': False, 'error': 'Policy version required'}), 400
            
            # Record consent
            consent = {
                'user_id': user_id,
                'policy_version': data['version'],
                'consented_at': int(time.time()),
                'ip_address': request.remote_addr
            }
            
            self._save_policy_consent(user_id, consent)
            
            # Log consent
            self._log_security_event(user_id, 'policy_consent', f'Consented to privacy policy version {data["version"]}')
            
            return jsonify({
                'success': True,
                'message': 'Consent recorded successfully'
            })
    
    def _register_jwt_callbacks(self):
        """Register JWT callbacks for additional security checks"""
        
        @self.jwt.token_in_blocklist_loader
        def check_if_token_revoked(jwt_header, jwt_payload):
            """Check if token has been revoked"""
            jti = jwt_payload["jti"]
            token_in_blocklist = self._is_token_revoked(jti)
            return token_in_blocklist
        
        @self.jwt.user_lookup_loader
        def user_lookup_callback(_jwt_header, jwt_data):
            """Load user from JWT data"""
            identity = jwt_data["sub"]
            return self._get_user_by_id(identity)
    
    def encrypt_data(self, data: Any) -> str:
        """
        Encrypt data using the master key
        
        Args:
            data: Data to encrypt (will be converted to JSON)
            
        Returns:
            Base64-encoded encrypted data
        """
        # Convert data to JSON string
        data_json = json.dumps(data)
        
        # Encrypt the data
        encrypted_data = self.cipher.encrypt(data_json.encode())
        
        # Return as base64 string
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Any:
        """
        Decrypt encrypted data
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data (parsed from JSON)
        """
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        # Decrypt the data
        decrypted_data = self.cipher.decrypt(encrypted_bytes).decode()
        
        # Parse JSON
        return json.loads(decrypted_data)
    
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive an encryption key from a password
        
        Args:
            password: Password to derive key from
            salt: Optional salt, will be generated if not provided
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA recovery"""
        codes = []
        for _ in range(count):
            # Generate a random 8-character code
            code = ''.join(secrets.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
            codes.append(code)
        return codes
    
    def _save_mfa_data(self, user_id: str, mfa_data: Dict[str, Any]) -> None:
        """Save MFA data to file"""
        mfa_path = os.path.join(self.mfa_dir, f"{user_id}.json")
        with open(mfa_path, 'w') as f:
            json.dump(mfa_data, f, indent=2)
    
    def _get_mfa_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get MFA data for a user"""
        mfa_path = os.path.join(self.mfa_dir, f"{user_id}.json")
        if os.path.exists(mfa_path):
            with open(mfa_path, 'r') as f:
                return json.load(f)
        return None
    
    def _log_security_event(self, user_id: str, event_type: str, description: str, metadata: Dict[str, Any] = None) -> None:
        """Log a security event"""
        # Create log entry
        log_entry = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'event_type': event_type,
            'description': description,
            'timestamp': int(time.time()),
            'ip_address': request.remote_addr if request else 'unknown',
            'user_agent': request.user_agent.string if request and request.user_agent else 'unknown',
            'metadata': metadata or {}
        }
        
        # Save log entry
        log_date = datetime.fromtimestamp(log_entry['timestamp']).strftime('%Y-%m-%d')
        log_dir = os.path.join(self.logs_dir, log_date)
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, f"{log_entry['id']}.json")
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Also save to user-specific log file for quick access
        user_log_dir = os.path.join(self.logs_dir, 'users', user_id)
        os.makedirs(user_log_dir, exist_ok=True)
        
        user_log_path = os.path.join(user_log_dir, f"{log_entry['id']}.json")
        with open(user_log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
    
    def _get_security_logs(self, user_id: str, event_type: Optional[str] = None, 
                          start_time: Optional[int] = None, end_time: Optional[int] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get security logs for a user"""
        logs = []
        
        # Get user-specific logs
        user_log_dir = os.path.join(self.logs_dir, 'users', user_id)
        if os.path.exists(user_log_dir):
            for filename in os.listdir(user_log_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(user_log_dir, filename), 'r') as f:
                        log = json.load(f)
                        
                        # Apply filters
                        if event_type and log['event_type'] != event_type:
                            continue
                        
                        if start_time and log['timestamp'] < start_time:
                            continue
                        
                        if end_time and log['timestamp'] > end_time:
                            continue
                        
                        logs.append(log)
                        
                        if len(logs) >= limit:
                            break
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return logs[:limit]
    
    def _save_privacy_policy(self, policy: Dict[str, Any]) -> None:
        """Save privacy policy to file"""
        policy_path = os.path.join(self.policies_dir, f"policy_v{policy['version']}.json")
        with open(policy_path, 'w') as f:
            json.dump(policy, f, indent=2)
    
    def _get_latest_privacy_policy(self) -> Optional[Dict[str, Any]]:
        """Get the latest privacy policy"""
        latest_version = 0
        latest_policy = None
        
        for filename in os.listdir(self.policies_dir):
            if filename.startswith('policy_v') and filename.endswith('.json'):
                try:
                    version = int(filename[8:-5])  # Extract version number from filename
                    if version > latest_version:
                        with open(os.path.join(self.policies_dir, filename), 'r') as f:
                            policy = json.load(f)
                            latest_version = version
                            latest_policy = policy
                except (ValueError, IndexError):
                    continue
        
        return latest_policy
    
    def _get_next_policy_version(self) -> int:
        """Get the next policy version number"""
        latest_policy = self._get_latest_privacy_policy()
        if latest_policy:
            return latest_policy['version'] + 1
        return 1
    
    def _save_policy_consent(self, user_id: str, consent: Dict[str, Any]) -> None:
        """Save user consent to privacy policy"""
        consent_dir = os.path.join(self.policies_dir, 'consents')
        os.makedirs(consent_dir, exist_ok=True)
        
        consent_path = os.path.join(consent_dir, f"{user_id}_v{consent['policy_version']}.json")
        with open(consent_path, 'w') as f:
            json.dump(consent, f, indent=2)
    
    def _is_token_revoked(self, jti: str) -> bool:
        """Check if a token has been revoked"""
        # In a real implementation, you would check against a database of revoked tokens
        # This is a simplified implementation
        revoked_tokens_path = os.path.join(self.security_dir, 'revoked_tokens.json')
        
        if os.path.exists(revoked_tokens_path):
            with open(revoked_tokens_path, 'r') as f:
                revoked_tokens = json.load(f)
                return jti in revoked_tokens
        
        return False
    
    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID"""
        # In a real implementation, this would be handled by a user service
        # This is a simplified implementation
        users_dir = os.path.join(self.config.get('data_dir', 'data'), 'users')
        user_path = os.path.join(users_dir, f"{user_id}.json")
        
        if os.path.exists(user_path):
            with open(user_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _get_user_email(self, user_id: str) -> str:
        """Get a user's email address"""
        user = self._get_user_by_id(user_id)
        if user:
            return user.get('email', f"user_{user_id}")
        return f"user_{user_id}"
    
    def _verify_user_password(self, user_id: str, password: str) -> bool:
        """Verify a user's password"""
        # In a real implementation, this would be handled by a user service
        # This is a simplified implementation
        user = self._get_user_by_id(user_id)
        
        if not user:
            return False
        
        # Hash the password and compare with stored hash
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == user.get('password_hash')
    
    def _is_admin_user(self, user_id: str) -> bool:
        """Check if a user has admin role"""
        # In a real implementation, this would be handled by a user service
        # This is a simplified implementation
        user = self._get_user_by_id(user_id)
        
        if not user:
            return False
        
        return user.get('role') == 'admin'

# Function to initialize and register the security features with the Flask app
def init_security_features(app: Flask, config: Dict[str, Any]) -> SecurityManager:
    """Initialize and register security features with the Flask app"""
    security_manager = SecurityManager(app, config)
    return security_manager
