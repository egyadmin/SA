"""
Adaptive Learning Module for Manus Clone

This module implements adaptive learning features including:
- Learning from user interactions
- Experience customization based on user behavior
- Self-improvement capabilities
- Continuous feedback system
"""

import os
import json
import time
import logging
import uuid
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveLearningManager:
    """Manages adaptive learning features for the Manus Clone application"""
    
    def __init__(self, app: Flask, config: Dict[str, Any], analytics_manager=None):
        """
        Initialize the Adaptive Learning Manager with configuration
        
        Args:
            app: Flask application instance
            config: Dictionary containing configuration parameters
            analytics_manager: Optional reference to the Analytics Manager for data access
        """
        self.config = config
        self.app = app
        self.analytics_manager = analytics_manager
        
        # Initialize storage directories
        self.adaptive_dir = os.path.join(config.get('data_dir', 'data'), 'adaptive_learning')
        os.makedirs(self.adaptive_dir, exist_ok=True)
        
        # User models storage
        self.user_models_dir = os.path.join(self.adaptive_dir, 'user_models')
        os.makedirs(self.user_models_dir, exist_ok=True)
        
        # Behavior patterns storage
        self.patterns_dir = os.path.join(self.adaptive_dir, 'patterns')
        os.makedirs(self.patterns_dir, exist_ok=True)
        
        # Feedback storage
        self.feedback_dir = os.path.join(self.adaptive_dir, 'feedback')
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Self-improvement data storage
        self.improvement_dir = os.path.join(self.adaptive_dir, 'improvement')
        os.makedirs(self.improvement_dir, exist_ok=True)
        
        # Initialize learning models
        self._init_learning_models()
        
        # Register routes
        self._register_routes()
        
        # Schedule periodic learning tasks
        self._schedule_learning_tasks()
        
        logger.info("Adaptive Learning Manager initialized successfully")
    
    def _init_learning_models(self):
        """Initialize learning models"""
        # User behavior clustering model
        self.behavior_cluster_model = None
        
        # User preference prediction model
        self.preference_models = {}
        
        # Content relevance model
        self.content_relevance_model = None
        
        # Feature usage prediction model
        self.feature_usage_model = None
        
        # Load existing models if available
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing learning models if available"""
        try:
            # Load behavior clustering model
            cluster_model_path = os.path.join(self.adaptive_dir, 'behavior_cluster_model.pkl')
            if os.path.exists(cluster_model_path):
                import pickle
                with open(cluster_model_path, 'rb') as f:
                    self.behavior_cluster_model = pickle.load(f)
                logger.info("Loaded behavior clustering model")
            
            # Load content relevance model
            relevance_model_path = os.path.join(self.adaptive_dir, 'content_relevance_model.pkl')
            if os.path.exists(relevance_model_path):
                import pickle
                with open(relevance_model_path, 'rb') as f:
                    self.content_relevance_model = pickle.load(f)
                logger.info("Loaded content relevance model")
            
            # Load feature usage model
            feature_model_path = os.path.join(self.adaptive_dir, 'feature_usage_model.pkl')
            if os.path.exists(feature_model_path):
                import pickle
                with open(feature_model_path, 'rb') as f:
                    self.feature_usage_model = pickle.load(f)
                logger.info("Loaded feature usage model")
            
            # Load user preference models
            for filename in os.listdir(self.user_models_dir):
                if filename.endswith('.pkl') and filename.startswith('pref_'):
                    user_id = filename[5:-4]  # Extract user_id from filename
                    model_path = os.path.join(self.user_models_dir, filename)
                    
                    import pickle
                    with open(model_path, 'rb') as f:
                        self.preference_models[user_id] = pickle.load(f)
                    
                    logger.info(f"Loaded preference model for user {user_id}")
        
        except Exception as e:
            logger.error(f"Error loading existing models: {str(e)}")
    
    def _register_routes(self):
        """Register HTTP routes for adaptive learning features"""
        
        @self.app.route('/api/adaptive/profile', methods=['GET'])
        def get_user_adaptive_profile():
            """Get adaptive profile for a user"""
            user_id = request.args.get('user_id')
            
            if not user_id:
                return jsonify({'success': False, 'error': 'User ID is required'}), 400
            
            # Get user's adaptive profile
            profile = self._get_user_adaptive_profile(user_id)
            
            return jsonify({
                'success': True,
                'profile': profile
            })
        
        @self.app.route('/api/adaptive/recommendations', methods=['GET'])
        def get_adaptive_recommendations():
            """Get personalized recommendations based on adaptive learning"""
            user_id = request.args.get('user_id')
            category = request.args.get('category', 'all')
            limit = int(request.args.get('limit', 5))
            
            if not user_id:
                return jsonify({'success': False, 'error': 'User ID is required'}), 400
            
            # Get adaptive recommendations
            recommendations = self._get_adaptive_recommendations(user_id, category, limit)
            
            return jsonify({
                'success': True,
                'recommendations': recommendations
            })
        
        @self.app.route('/api/adaptive/feedback', methods=['POST'])
        def submit_adaptive_feedback():
            """Submit feedback for adaptive learning"""
            data = request.get_json()
            
            if 'user_id' not in data or 'item_id' not in data or 'rating' not in data:
                return jsonify({'success': False, 'error': 'User ID, item ID, and rating are required'}), 400
            
            user_id = data['user_id']
            item_id = data['item_id']
            rating = data['rating']
            item_type = data.get('item_type', 'content')
            feedback_text = data.get('feedback', '')
            
            # Record feedback
            feedback_id = self._record_feedback(user_id, item_id, item_type, rating, feedback_text)
            
            # Update user model with new feedback
            self._update_user_model_with_feedback(user_id, item_id, item_type, rating)
            
            return jsonify({
                'success': True,
                'feedback_id': feedback_id,
                'message': 'Feedback recorded successfully'
            })
        
        @self.app.route('/api/adaptive/customize', methods=['GET'])
        def get_customized_experience():
            """Get customized experience settings for a user"""
            user_id = request.args.get('user_id')
            
            if not user_id:
                return jsonify({'success': False, 'error': 'User ID is required'}), 400
            
            # Get customized experience settings
            settings = self._get_customized_experience(user_id)
            
            return jsonify({
                'success': True,
                'settings': settings
            })
        
        @self.app.route('/api/adaptive/customize', methods=['POST'])
        def update_customized_experience():
            """Update customized experience settings for a user"""
            data = request.get_json()
            
            if 'user_id' not in data or 'settings' not in data:
                return jsonify({'success': False, 'error': 'User ID and settings are required'}), 400
            
            user_id = data['user_id']
            settings = data['settings']
            
            # Update customized experience settings
            success = self._update_customized_experience(user_id, settings)
            
            return jsonify({
                'success': success,
                'message': 'Customized experience settings updated successfully' if success else 'Failed to update settings'
            })
        
        @self.app.route('/api/adaptive/track', methods=['POST'])
        def track_user_interaction():
            """Track user interaction for adaptive learning"""
            data = request.get_json()
            
            if 'user_id' not in data or 'interaction_type' not in data:
                return jsonify({'success': False, 'error': 'User ID and interaction type are required'}), 400
            
            user_id = data['user_id']
            interaction_type = data['interaction_type']
            interaction_data = data.get('data', {})
            
            # Track interaction
            interaction_id = self._track_interaction(user_id, interaction_type, interaction_data)
            
            return jsonify({
                'success': True,
                'interaction_id': interaction_id,
                'message': 'Interaction tracked successfully'
            })
        
        @self.app.route('/api/adaptive/insights', methods=['GET'])
        def get_adaptive_insights():
            """Get insights from adaptive learning"""
            user_id = request.args.get('user_id')
            
            # Get adaptive insights
            insights = self._get_adaptive_insights(user_id)
            
            return jsonify({
                'success': True,
                'insights': insights
            })
    
    def _schedule_learning_tasks(self):
        """Schedule periodic learning tasks"""
        # In a real implementation, you would use a task scheduler like Celery
        # For this example, we'll just set up the structure
        
        # Define learning tasks
        learning_tasks = [
            {
                'name': 'update_behavior_clusters',
                'interval': 86400,  # Daily
                'last_run': None
            },
            {
                'name': 'update_content_relevance_model',
                'interval': 86400 * 3,  # Every 3 days
                'last_run': None
            },
            {
                'name': 'update_feature_usage_model',
                'interval': 86400 * 7,  # Weekly
                'last_run': None
            },
            {
                'name': 'analyze_feedback_patterns',
                'interval': 86400 * 2,  # Every 2 days
                'last_run': None
            },
            {
                'name': 'generate_improvement_suggestions',
                'interval': 86400 * 14,  # Every 2 weeks
                'last_run': None
            }
        ]
        
        # Save learning tasks configuration
        tasks_path = os.path.join(self.adaptive_dir, 'learning_tasks.json')
        with open(tasks_path, 'w') as f:
            json.dump(learning_tasks, f, indent=2)
        
        logger.info("Learning tasks scheduled")
    
    def _get_user_adaptive_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get adaptive profile for a user
        
        Args:
            user_id: User ID
            
        Returns:
            User's adaptive profile
        """
        # Check if user profile exists
        profile_path = os.path.join(self.user_models_dir, f"profile_{user_id}.json")
        
        if os.path.exists(profile_path):
            # Load existing profile
            with open(profile_path, 'r') as f:
                profile = json.load(f)
        else:
            # Create new profile
            profile = self._create_new_user_profile(user_id)
            
            # Save new profile
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
        
        return profile
    
    def _create_new_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Create a new adaptive profile for a user
        
        Args:
            user_id: User ID
            
        Returns:
            New user profile
        """
        # Create basic profile with default values
        profile = {
            'user_id': user_id,
            'created_at': int(time.time()),
            'last_updated': int(time.time()),
            'behavior_cluster': None,
            'preferences': {
                'content_types': {},
                'features': {},
                'ui': {
                    'theme': 'light',
                    'layout': 'default',
                    'font_size': 'medium'
                },
                'notification_frequency': 'medium'
            },
            'usage_patterns': {
                'active_hours': [],
                'session_duration': 0,
                'frequent_actions': [],
                'feature_usage': {}
            },
            'learning_progress': {
                'completed_tutorials': [],
                'skill_levels': {},
                'badges': []
            }
        }
        
        return profile
    
    def _get_adaptive_recommendations(self, user_id: str, category: str = 'all', limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get personalized recommendations based on adaptive learning
        
        Args:
            user_id: User ID
            category: Recommendation category (all, content, features, learning)
            limit: Maximum number of recommendations per category
            
        Returns:
            Dictionary of recommendations by category
        """
        # Get user profile
        profile = self._get_user_adaptive_profile(user_id)
        
        recommendations = {}
        
        # Get content recommendations if requested
        if category in ['all', 'content']:
            content_recs = self._get_content_recommendations(user_id, profile, limit)
            recommendations['content'] = content_recs
        
        # Get feature recommendations if requested
        if category in ['all', 'features']:
            feature_recs = self._get_feature_recommendations(user_id, profile, limit)
            recommendations['features'] = feature_recs
        
        # Get learning recommendations if requested
        if category in ['all', 'learning']:
            learning_recs = self._get_learning_recommendations(user_id, profile, limit)
            recommendations['learning'] = learning_recs
        
        # Get UI customization recommendations if requested
        if category in ['all', 'ui']:
            ui_recs = self._get_ui_recommendations(user_id, profile, limit)
            recommendations['ui'] = ui_recs
        
        return recommendations
    
    def _get_content_recommendations(self, user_id: str, profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Get content recommendations for a user
        
        Args:
            user_id: User ID
            profile: User's adaptive profile
            limit: Maximum number of recommendations
            
        Returns:
            List of content recommendations
        """
        # This is a simplified implementation
        # In a real system, you would use a recommendation algorithm based on user preferences and behavior
        
        # Get user's content preferences
        content_preferences = profile.get('preferences', {}).get('content_types', {})
        
        # Sort content types by preference score
        preferred_types = sorted(content_preferences.items(), key=lambda x: x[1], reverse=True)
        
        # Generate recommendations based on preferred content types
        recommendations = []
        
        # If we have the analytics manager, use it to get popular content
        if self.analytics_manager:
            # Use analytics to get popular content
            popular_content = self.analytics_manager._get_popular_content_recommendations(limit * 2)
            
            # Filter and rank based on user preferences
            for content in popular_content:
                content_type = content.get('type')
                
                # Calculate relevance score based on user preferences
                relevance_score = 0.5  # Default score
                
                for pref_type, pref_score in preferred_types:
                    if content_type == pref_type:
                        relevance_score = pref_score
                        break
                
                # Add relevance score to content
                content['relevance_score'] = relevance_score
            
            # Sort by relevance score
            popular_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Take top recommendations
            recommendations = popular_content[:limit]
        else:
            # Generate synthetic recommendations
            for i in range(min(limit, len(preferred_types))):
                content_type, preference_score = preferred_types[i]
                
                recommendations.append({
                    'content_id': f"rec_content_{i}",
                    'title': f"Recommended {content_type.capitalize()} Content",
                    'type': content_type,
                    'relevance_score': preference_score,
                    'reason': f"Based on your preference for {content_type} content"
                })
        
        return recommendations
    
    def _get_feature_recommendations(self, user_id: str, profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Get feature recommendations for a user
        
        Args:
            user_id: User ID
            profile: User's adaptive profile
            limit: Maximum number of recommendations
            
        Returns:
            List of feature recommendations
        """
        # Get user's feature usage patterns
        feature_usage = profile.get('usage_patterns', {}).get('feature_usage', {})
        
        # Get all available features (in a real system, this would come from a feature registry)
        available_features = [
            {'id': 'collaborative_editing', 'name': 'Collaborative Editing', 'category': 'collaboration'},
            {'id': 'ai_image_generation', 'name': 'AI Image Generation', 'category': 'generative_ai'},
            {'id': 'advanced_analytics', 'name': 'Advanced Analytics Dashboard', 'category': 'analytics'},
            {'id': 'custom_templates', 'name': 'Custom Templates', 'category': 'content'},
            {'id': 'voice_commands', 'name': 'Voice Commands', 'category': 'accessibility'},
            {'id': 'data_visualization', 'name': 'Data Visualization Tools', 'category': 'analytics'},
            {'id': 'code_generation', 'name': 'Code Generation', 'category': 'generative_ai'},
            {'id': 'automated_workflows', 'name': 'Automated Workflows', 'category': 'automation'}
        ]
        
        # Filter out features the user already uses frequently
        frequently_used = set(feature_id for feature_id, usage in feature_usage.items() if usage > 0.7)
        
        # Filter features
        candidate_features = [f for f in available_features if f['id'] not in frequently_used]
        
        # Score features based on user's preferences and behavior cluster
        scored_features = []
        
        for feature in candidate_features:
            # Base score
            score = 0.5
            
            # Adjust score based on user's preferences
            feature_preferences = profile.get('preferences', {}).get('features', {})
            if feature['category'] in feature_preferences:
                score += feature_preferences[feature['category']] * 0.3
            
            # Adjust score based on behavior cluster
            behavior_cluster = profile.get('behavior_cluster')
            if behavior_cluster is not None:
                # In a real implementation, you would have cluster-based recommendations
                if behavior_cluster == 0 and feature['category'] == 'analytics':
                    score += 0.2
                elif behavior_cluster == 1 and feature['category'] == 'generative_ai':
                    score += 0.2
                elif behavior_cluster == 2 and feature['category'] == 'collaboration':
                    score += 0.2
            
            scored_features.append((feature, score))
        
        # Sort by score
        scored_features.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        
        for feature, score in scored_features[:limit]:
            recommendations.append({
                'feature_id': feature['id'],
                'name': feature['name'],
                'category': feature['category'],
                'relevance_score': score,
                'reason': f"This feature matches your usage patterns and preferences"
            })
        
        return recommendations
    
    def _get_learning_recommendations(self, user_id: str, profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Get learning path recommendations for a user
        
        Args:
            user_id: User ID
            profile: User's adaptive profile
            limit: Maximum number of recommendations
            
        Returns:
            List of learning recommendations
        """
        # Get user's learning progress
        learning_progress = profile.get('learning_progress', {})
        completed_tutorials = set(learning_progress.get('completed_tutorials', []))
        skill_levels = learning_progress.get('skill_levels', {})
        
        # Define available tutorials (in a real system, this would come from a content database)
        available_tutorials = [
            {'id': 'getting_started', 'name': 'Getting Started Guide', 'difficulty': 'beginner', 'skill': 'basics'},
            {'id': 'collaborative_features', 'name': 'Using Collaborative Features', 'difficulty': 'intermediate', 'skill': 'collaboration'},
            {'id': 'ai_generation', 'name': 'AI Content Generation', 'difficulty': 'intermediate', 'skill': 'ai'},
            {'id': 'advanced_analytics', 'name': 'Advanced Analytics', 'difficulty': 'advanced', 'skill': 'analytics'},
            {'id': 'security_features', 'name': 'Security and Privacy Features', 'difficulty': 'intermediate', 'skill': 'security'},
            {'id': 'custom_workflows', 'name': 'Creating Custom Workflows', 'difficulty': 'advanced', 'skill': 'automation'},
            {'id': 'api_integration', 'name': 'API Integration Guide', 'difficulty': 'advanced', 'skill': 'development'},
            {'id': 'mobile_features', 'name': 'Mobile App Features', 'difficulty': 'beginner', 'skill': 'mobile'}
        ]
        
        # Filter out completed tutorials
        candidate_tutorials = [t for t in available_tutorials if t['id'] not in completed_tutorials]
        
        # Score tutorials based on user's skill levels and preferences
        scored_tutorials = []
        
        for tutorial in candidate_tutorials:
            # Base score
            score = 0.5
            
            # Adjust score based on skill level
            skill = tutorial['skill']
            skill_level = skill_levels.get(skill, 0)
            
            # Recommend tutorials that match the user's skill level
            if tutorial['difficulty'] == 'beginner' and skill_level < 0.3:
                score += 0.3
            elif tutorial['difficulty'] == 'intermediate' and 0.3 <= skill_level < 0.7:
                score += 0.3
            elif tutorial['difficulty'] == 'advanced' and skill_level >= 0.7:
                score += 0.3
            
            # Adjust score based on feature preferences
            feature_preferences = profile.get('preferences', {}).get('features', {})
            if skill in feature_preferences:
                score += feature_preferences[skill] * 0.2
            
            scored_tutorials.append((tutorial, score))
        
        # Sort by score
        scored_tutorials.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        
        for tutorial, score in scored_tutorials[:limit]:
            recommendations.append({
                'tutorial_id': tutorial['id'],
                'name': tutorial['name'],
                'difficulty': tutorial['difficulty'],
                'skill': tutorial['skill'],
                'relevance_score': score,
                'reason': f"This tutorial matches your current skill level in {tutorial['skill']}"
            })
        
        return recommendations
    
    def _get_ui_recommendations(self, user_id: str, profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Get UI customization recommendations for a user
        
        Args:
            user_id: User ID
            profile: User's adaptive profile
            limit: Maximum number of recommendations
            
        Returns:
            List of UI recommendations
        """
        # Get user's current UI settings
        ui_preferences = profile.get('preferences', {}).get('ui', {})
        current_theme = ui_preferences.get('theme', 'light')
        current_layout = ui_preferences.get('layout', 'default')
        current_font_size = ui_preferences.get('font_size', 'medium')
        
        # Get usage patterns
        usage_patterns = profile.get('usage_patterns', {})
        active_hours = usage_patterns.get('active_hours', [])
        session_duration = usage_patterns.get('session_duration', 0)
        
        # Generate UI recommendations based on patterns
        recommendations = []
        
        # Theme recommendation based on active hours
        if active_hours and len(active_hours) > 0:
            avg_hour = sum(active_hours) / len(active_hours)
            if avg_hour < 6 or avg_hour > 20:  # Early morning or late night
                if current_theme != 'dark':
                    recommendations.append({
                        'setting_id': 'theme',
                        'name': 'Dark Theme',
                        'current_value': current_theme,
                        'recommended_value': 'dark',
                        'relevance_score': 0.8,
                        'reason': 'Based on your usage during evening/night hours'
                    })
            else:
                if current_theme != 'light':
                    recommendations.append({
                        'setting_id': 'theme',
                        'name': 'Light Theme',
                        'current_value': current_theme,
                        'recommended_value': 'light',
                        'relevance_score': 0.7,
                        'reason': 'Based on your usage during daylight hours'
                    })
        
        # Layout recommendation based on session duration
        if session_duration > 3600:  # Long sessions (> 1 hour)
            if current_layout != 'expanded':
                recommendations.append({
                    'setting_id': 'layout',
                    'name': 'Expanded Layout',
                    'current_value': current_layout,
                    'recommended_value': 'expanded',
                    'relevance_score': 0.75,
                    'reason': 'Optimized for your longer work sessions'
                })
        elif session_duration < 600:  # Short sessions (< 10 minutes)
            if current_layout != 'compact':
                recommendations.append({
                    'setting_id': 'layout',
                    'name': 'Compact Layout',
                    'current_value': current_layout,
                    'recommended_value': 'compact',
                    'relevance_score': 0.75,
                    'reason': 'Optimized for your quick check-in sessions'
                })
        
        # Font size recommendation based on behavior cluster
        behavior_cluster = profile.get('behavior_cluster')
        if behavior_cluster is not None:
            if behavior_cluster == 0 and current_font_size != 'small':  # Data-focused users
                recommendations.append({
                    'setting_id': 'font_size',
                    'name': 'Smaller Font Size',
                    'current_value': current_font_size,
                    'recommended_value': 'small',
                    'relevance_score': 0.7,
                    'reason': 'Allows more data to be visible at once'
                })
            elif behavior_cluster == 2 and current_font_size != 'large':  # Content-focused users
                recommendations.append({
                    'setting_id': 'font_size',
                    'name': 'Larger Font Size',
                    'current_value': current_font_size,
                    'recommended_value': 'large',
                    'relevance_score': 0.7,
                    'reason': 'Improves readability for content creation'
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return recommendations[:limit]
    
    def _record_feedback(self, user_id: str, item_id: str, item_type: str, rating: float, feedback_text: str = '') -> str:
        """
        Record user feedback for adaptive learning
        
        Args:
            user_id: User ID
            item_id: Item ID (content, feature, etc.)
            item_type: Type of item (content, feature, ui, etc.)
            rating: Rating (0-5)
            feedback_text: Optional feedback text
            
        Returns:
            Feedback ID
        """
        # Create feedback record
        feedback_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        feedback = {
            'id': feedback_id,
            'user_id': user_id,
            'item_id': item_id,
            'item_type': item_type,
            'rating': rating,
            'feedback_text': feedback_text,
            'timestamp': timestamp,
            'date': datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
            'time': datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        }
        
        # Save feedback to appropriate directory
        user_feedback_dir = os.path.join(self.feedback_dir, user_id)
        os.makedirs(user_feedback_dir, exist_ok=True)
        
        feedback_path = os.path.join(user_feedback_dir, f"{feedback_id}.json")
        with open(feedback_path, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        # Also save to item-specific directory
        item_feedback_dir = os.path.join(self.feedback_dir, 'items', item_type)
        os.makedirs(item_feedback_dir, exist_ok=True)
        
        item_feedback_path = os.path.join(item_feedback_dir, f"{item_id}_{feedback_id}.json")
        with open(item_feedback_path, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        return feedback_id
    
    def _update_user_model_with_feedback(self, user_id: str, item_id: str, item_type: str, rating: float) -> None:
        """
        Update user model with new feedback
        
        Args:
            user_id: User ID
            item_id: Item ID
            item_type: Type of item
            rating: Rating
        """
        # Get user profile
        profile = self._get_user_adaptive_profile(user_id)
        
        # Normalize rating to 0-1 scale
        normalized_rating = rating / 5.0
        
        # Update preferences based on item type
        if item_type == 'content':
            # Get content type (in a real system, you would look this up)
            content_type = item_id.split('_')[0] if '_' in item_id else 'unknown'
            
            # Update content type preference
            preferences = profile.get('preferences', {})
            content_preferences = preferences.get('content_types', {})
            
            if content_type in content_preferences:
                # Update existing preference with exponential moving average
                alpha = 0.3  # Weight for new observation
                content_preferences[content_type] = (1 - alpha) * content_preferences[content_type] + alpha * normalized_rating
            else:
                # Add new preference
                content_preferences[content_type] = normalized_rating
            
            preferences['content_types'] = content_preferences
            profile['preferences'] = preferences
        
        elif item_type == 'feature':
            # Get feature category (in a real system, you would look this up)
            feature_category = item_id.split('_')[0] if '_' in item_id else 'unknown'
            
            # Update feature preference
            preferences = profile.get('preferences', {})
            feature_preferences = preferences.get('features', {})
            
            if feature_category in feature_preferences:
                # Update existing preference with exponential moving average
                alpha = 0.3  # Weight for new observation
                feature_preferences[feature_category] = (1 - alpha) * feature_preferences[feature_category] + alpha * normalized_rating
            else:
                # Add new preference
                feature_preferences[feature_category] = normalized_rating
            
            preferences['features'] = feature_preferences
            profile['preferences'] = preferences
        
        elif item_type == 'ui':
            # Update UI preference (no specific action needed here)
            pass
        
        # Update last updated timestamp
        profile['last_updated'] = int(time.time())
        
        # Save updated profile
        profile_path = os.path.join(self.user_models_dir, f"profile_{user_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def _get_customized_experience(self, user_id: str) -> Dict[str, Any]:
        """
        Get customized experience settings for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Customized experience settings
        """
        # Get user profile
        profile = self._get_user_adaptive_profile(user_id)
        
        # Extract relevant settings
        ui_preferences = profile.get('preferences', {}).get('ui', {})
        notification_frequency = profile.get('preferences', {}).get('notification_frequency', 'medium')
        
        # Get behavior-based customizations
        behavior_cluster = profile.get('behavior_cluster')
        
        # Default customizations
        customizations = {
            'ui': ui_preferences,
            'notifications': {
                'frequency': notification_frequency
            },
            'content_display': {
                'default_sort': 'relevance',
                'items_per_page': 20
            },
            'features': {
                'quick_access': []
            }
        }
        
        # Customize based on behavior cluster
        if behavior_cluster is not None:
            if behavior_cluster == 0:  # Data-focused users
                customizations['content_display']['default_sort'] = 'date'
                customizations['content_display']['items_per_page'] = 50
                customizations['features']['quick_access'] = ['analytics', 'data_visualization', 'export']
            
            elif behavior_cluster == 1:  # Creative users
                customizations['content_display']['default_sort'] = 'trending'
                customizations['features']['quick_access'] = ['ai_generation', 'templates', 'media_library']
            
            elif behavior_cluster == 2:  # Collaborative users
                customizations['content_display']['default_sort'] = 'activity'
                customizations['features']['quick_access'] = ['share', 'comments', 'collaborative_editing']
        
        # Get feature usage to customize quick access
        feature_usage = profile.get('usage_patterns', {}).get('feature_usage', {})
        
        # Add most used features to quick access if not already there
        if feature_usage:
            # Sort features by usage
            sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
            
            # Add top features to quick access
            for feature, _ in sorted_features[:3]:
                if feature not in customizations['features']['quick_access']:
                    customizations['features']['quick_access'].append(feature)
        
        return customizations
    
    def _update_customized_experience(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """
        Update customized experience settings for a user
        
        Args:
            user_id: User ID
            settings: New settings
            
        Returns:
            Success flag
        """
        try:
            # Get user profile
            profile = self._get_user_adaptive_profile(user_id)
            
            # Update UI preferences
            if 'ui' in settings:
                profile.setdefault('preferences', {}).setdefault('ui', {}).update(settings['ui'])
            
            # Update notification preferences
            if 'notifications' in settings and 'frequency' in settings['notifications']:
                profile.setdefault('preferences', {})['notification_frequency'] = settings['notifications']['frequency']
            
            # Update content display preferences
            if 'content_display' in settings:
                profile.setdefault('preferences', {}).setdefault('content_display', {}).update(settings['content_display'])
            
            # Update last updated timestamp
            profile['last_updated'] = int(time.time())
            
            # Save updated profile
            profile_path = os.path.join(self.user_models_dir, f"profile_{user_id}.json")
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error updating customized experience: {str(e)}")
            return False
    
    def _track_interaction(self, user_id: str, interaction_type: str, interaction_data: Dict[str, Any]) -> str:
        """
        Track user interaction for adaptive learning
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            interaction_data: Interaction data
            
        Returns:
            Interaction ID
        """
        # Create interaction record
        interaction_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        interaction = {
            'id': interaction_id,
            'user_id': user_id,
            'type': interaction_type,
            'data': interaction_data,
            'timestamp': timestamp,
            'date': datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
            'time': datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        }
        
        # Save interaction to user's directory
        user_interactions_dir = os.path.join(self.adaptive_dir, 'interactions', user_id)
        os.makedirs(user_interactions_dir, exist_ok=True)
        
        interaction_path = os.path.join(user_interactions_dir, f"{interaction_id}.json")
        with open(interaction_path, 'w') as f:
            json.dump(interaction, f, indent=2)
        
        # Update user profile based on interaction
        self._update_profile_from_interaction(user_id, interaction_type, interaction_data)
        
        return interaction_id
    
    def _update_profile_from_interaction(self, user_id: str, interaction_type: str, interaction_data: Dict[str, Any]) -> None:
        """
        Update user profile based on interaction
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            interaction_data: Interaction data
        """
        # Get user profile
        profile = self._get_user_adaptive_profile(user_id)
        
        # Update usage patterns
        usage_patterns = profile.get('usage_patterns', {})
        
        # Update active hours
        if 'timestamp' in interaction_data:
            hour = datetime.datetime.fromtimestamp(interaction_data['timestamp']).hour
            active_hours = usage_patterns.get('active_hours', [])
            active_hours.append(hour)
            # Keep only the last 100 hours
            if len(active_hours) > 100:
                active_hours = active_hours[-100:]
            usage_patterns['active_hours'] = active_hours
        
        # Update session duration if session data is provided
        if 'session_duration' in interaction_data:
            session_duration = interaction_data['session_duration']
            # Update with exponential moving average
            alpha = 0.2  # Weight for new observation
            current_duration = usage_patterns.get('session_duration', 0)
            usage_patterns['session_duration'] = (1 - alpha) * current_duration + alpha * session_duration
        
        # Update feature usage if feature data is provided
        if interaction_type == 'feature_use' and 'feature_id' in interaction_data:
            feature_id = interaction_data['feature_id']
            feature_usage = usage_patterns.get('feature_usage', {})
            
            # Increment feature usage counter
            if feature_id in feature_usage:
                # Update with exponential moving average
                alpha = 0.1  # Weight for new observation
                feature_usage[feature_id] = min(1.0, (1 - alpha) * feature_usage[feature_id] + alpha)
            else:
                feature_usage[feature_id] = 0.1  # Initial usage value
            
            usage_patterns['feature_usage'] = feature_usage
        
        # Update frequent actions
        if interaction_type == 'action' and 'action_id' in interaction_data:
            action_id = interaction_data['action_id']
            frequent_actions = usage_patterns.get('frequent_actions', [])
            
            # Add action to list if not already there
            if action_id not in frequent_actions:
                frequent_actions.append(action_id)
                # Keep only the last 20 actions
                if len(frequent_actions) > 20:
                    frequent_actions = frequent_actions[-20:]
                usage_patterns['frequent_actions'] = frequent_actions
        
        # Update learning progress if learning data is provided
        if interaction_type == 'learning_complete' and 'tutorial_id' in interaction_data:
            tutorial_id = interaction_data['tutorial_id']
            learning_progress = profile.get('learning_progress', {})
            completed_tutorials = learning_progress.get('completed_tutorials', [])
            
            # Add tutorial to completed list if not already there
            if tutorial_id not in completed_tutorials:
                completed_tutorials.append(tutorial_id)
                learning_progress['completed_tutorials'] = completed_tutorials
            
            # Update skill level if provided
            if 'skill' in interaction_data and 'level' in interaction_data:
                skill = interaction_data['skill']
                level = interaction_data['level']
                skill_levels = learning_progress.get('skill_levels', {})
                
                # Update skill level
                skill_levels[skill] = level
                learning_progress['skill_levels'] = skill_levels
            
            profile['learning_progress'] = learning_progress
        
        # Update profile with modified usage patterns
        profile['usage_patterns'] = usage_patterns
        
        # Update last updated timestamp
        profile['last_updated'] = int(time.time())
        
        # Save updated profile
        profile_path = os.path.join(self.user_models_dir, f"profile_{user_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def _get_adaptive_insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get insights from adaptive learning
        
        Args:
            user_id: Optional user ID for user-specific insights
            
        Returns:
            Adaptive insights
        """
        insights = {}
        
        if user_id:
            # Get user-specific insights
            insights['user'] = self._get_user_insights(user_id)
        
        # Get system-wide insights
        insights['system'] = self._get_system_insights()
        
        # Get improvement suggestions
        insights['improvements'] = self._get_improvement_suggestions()
        
        return insights
    
    def _get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights for a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            User insights
        """
        # Get user profile
        profile = self._get_user_adaptive_profile(user_id)
        
        # Extract relevant data
        behavior_cluster = profile.get('behavior_cluster')
        usage_patterns = profile.get('usage_patterns', {})
        preferences = profile.get('preferences', {})
        learning_progress = profile.get('learning_progress', {})
        
        # Generate insights
        insights = {
            'user_id': user_id,
            'behavior_summary': self._get_behavior_summary(behavior_cluster),
            'usage_insights': self._analyze_usage_patterns(usage_patterns),
            'preference_insights': self._analyze_preferences(preferences),
            'learning_insights': self._analyze_learning_progress(learning_progress),
            'recommendations': {
                'features': self._get_feature_recommendations(user_id, profile, 3),
                'content': self._get_content_recommendations(user_id, profile, 3),
                'learning': self._get_learning_recommendations(user_id, profile, 3)
            }
        }
        
        return insights
    
    def _get_behavior_summary(self, behavior_cluster: Optional[int]) -> str:
        """
        Get a summary of user behavior based on cluster
        
        Args:
            behavior_cluster: Behavior cluster ID
            
        Returns:
            Behavior summary
        """
        if behavior_cluster is None:
            return "Insufficient data to determine behavior pattern"
        
        # Cluster descriptions (in a real system, these would be derived from data analysis)
        cluster_descriptions = {
            0: "Data-focused user who prioritizes analytics and information processing",
            1: "Creative user who focuses on content generation and design",
            2: "Collaborative user who emphasizes sharing and teamwork",
            3: "Task-oriented user who values efficiency and automation",
            4: "Learning-focused user who regularly engages with tutorials and documentation"
        }
        
        return cluster_descriptions.get(behavior_cluster, "Unknown behavior pattern")
    
    def _analyze_usage_patterns(self, usage_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user usage patterns
        
        Args:
            usage_patterns: Usage pattern data
            
        Returns:
            Usage pattern insights
        """
        insights = {}
        
        # Analyze active hours
        active_hours = usage_patterns.get('active_hours', [])
        if active_hours:
            avg_hour = sum(active_hours) / len(active_hours)
            if avg_hour < 6:
                time_pattern = "Early morning user"
            elif avg_hour < 12:
                time_pattern = "Morning user"
            elif avg_hour < 18:
                time_pattern = "Afternoon user"
            else:
                time_pattern = "Evening user"
            
            insights['time_pattern'] = time_pattern
        
        # Analyze session duration
        session_duration = usage_patterns.get('session_duration', 0)
        if session_duration < 300:  # Less than 5 minutes
            usage_style = "Brief check-ins"
        elif session_duration < 1800:  # Less than 30 minutes
            usage_style = "Short focused sessions"
        elif session_duration < 3600:  # Less than 1 hour
            usage_style = "Medium-length working sessions"
        else:
            usage_style = "Extended working sessions"
        
        insights['usage_style'] = usage_style
        
        # Analyze feature usage
        feature_usage = usage_patterns.get('feature_usage', {})
        if feature_usage:
            # Get top features
            top_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)[:3]
            insights['top_features'] = [feature for feature, _ in top_features]
            
            # Determine feature diversity
            feature_count = len(feature_usage)
            if feature_count < 5:
                feature_diversity = "Focused on a few key features"
            elif feature_count < 10:
                feature_diversity = "Uses a moderate range of features"
            else:
                feature_diversity = "Explores a wide range of features"
            
            insights['feature_diversity'] = feature_diversity
        
        return insights
    
    def _analyze_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user preferences
        
        Args:
            preferences: Preference data
            
        Returns:
            Preference insights
        """
        insights = {}
        
        # Analyze content type preferences
        content_preferences = preferences.get('content_types', {})
        if content_preferences:
            # Get top content types
            top_content = sorted(content_preferences.items(), key=lambda x: x[1], reverse=True)[:2]
            insights['preferred_content'] = [content for content, _ in top_content]
        
        # Analyze feature preferences
        feature_preferences = preferences.get('features', {})
        if feature_preferences:
            # Get top feature categories
            top_features = sorted(feature_preferences.items(), key=lambda x: x[1], reverse=True)[:2]
            insights['preferred_features'] = [feature for feature, _ in top_features]
        
        # Analyze UI preferences
        ui_preferences = preferences.get('ui', {})
        if ui_preferences:
            insights['ui_preferences'] = ui_preferences
        
        return insights
    
    def _analyze_learning_progress(self, learning_progress: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user learning progress
        
        Args:
            learning_progress: Learning progress data
            
        Returns:
            Learning progress insights
        """
        insights = {}
        
        # Analyze completed tutorials
        completed_tutorials = learning_progress.get('completed_tutorials', [])
        insights['completed_tutorial_count'] = len(completed_tutorials)
        
        # Analyze skill levels
        skill_levels = learning_progress.get('skill_levels', {})
        if skill_levels:
            # Get top skills
            top_skills = sorted(skill_levels.items(), key=lambda x: x[1], reverse=True)[:3]
            insights['top_skills'] = [skill for skill, _ in top_skills]
            
            # Calculate average skill level
            avg_skill_level = sum(skill_levels.values()) / len(skill_levels)
            
            if avg_skill_level < 0.3:
                proficiency = "Beginner"
            elif avg_skill_level < 0.7:
                proficiency = "Intermediate"
            else:
                proficiency = "Advanced"
            
            insights['proficiency_level'] = proficiency
        
        return insights
    
    def _get_system_insights(self) -> Dict[str, Any]:
        """
        Get system-wide insights from adaptive learning
        
        Returns:
            System insights
        """
        # This is a simplified implementation
        # In a real system, you would analyze data across all users
        
        # Get all user profiles
        user_profiles = self._get_all_user_profiles()
        
        # Count users by behavior cluster
        cluster_counts = {}
        for profile in user_profiles:
            cluster = profile.get('behavior_cluster')
            if cluster is not None:
                if cluster not in cluster_counts:
                    cluster_counts[cluster] = 0
                cluster_counts[cluster] += 1
        
        # Get popular content types
        content_type_scores = {}
        for profile in user_profiles:
            content_preferences = profile.get('preferences', {}).get('content_types', {})
            for content_type, score in content_preferences.items():
                if content_type not in content_type_scores:
                    content_type_scores[content_type] = []
                content_type_scores[content_type].append(score)
        
        # Calculate average scores
        avg_content_scores = {}
        for content_type, scores in content_type_scores.items():
            avg_content_scores[content_type] = sum(scores) / len(scores) if scores else 0
        
        # Get popular features
        feature_usage_counts = {}
        for profile in user_profiles:
            feature_usage = profile.get('usage_patterns', {}).get('feature_usage', {})
            for feature, usage in feature_usage.items():
                if feature not in feature_usage_counts:
                    feature_usage_counts[feature] = 0
                feature_usage_counts[feature] += 1
        
        # Compile insights
        insights = {
            'user_count': len(user_profiles),
            'behavior_clusters': cluster_counts,
            'popular_content_types': sorted(avg_content_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            'popular_features': sorted(feature_usage_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return insights
    
    def _get_all_user_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all user profiles
        
        Returns:
            List of user profiles
        """
        profiles = []
        
        for filename in os.listdir(self.user_models_dir):
            if filename.startswith('profile_') and filename.endswith('.json'):
                profile_path = os.path.join(self.user_models_dir, filename)
                
                try:
                    with open(profile_path, 'r') as f:
                        profile = json.load(f)
                        profiles.append(profile)
                except Exception as e:
                    logger.error(f"Error loading profile {filename}: {str(e)}")
        
        return profiles
    
    def _get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get system improvement suggestions based on adaptive learning
        
        Returns:
            List of improvement suggestions
        """
        # This is a simplified implementation
        # In a real system, you would analyze user feedback and behavior patterns
        
        # Get all feedback
        all_feedback = self._get_all_feedback()
        
        # Analyze feedback by item type
        feedback_by_type = {}
        for feedback in all_feedback:
            item_type = feedback.get('item_type')
            if item_type not in feedback_by_type:
                feedback_by_type[item_type] = []
            feedback_by_type[item_type].append(feedback)
        
        # Generate improvement suggestions
        suggestions = []
        
        # Analyze feature feedback
        if 'feature' in feedback_by_type:
            feature_feedback = feedback_by_type['feature']
            
            # Group by feature ID
            feedback_by_feature = {}
            for feedback in feature_feedback:
                feature_id = feedback.get('item_id')
                if feature_id not in feedback_by_feature:
                    feedback_by_feature[feature_id] = []
                feedback_by_feature[feature_id].append(feedback)
            
            # Find features with low ratings
            for feature_id, feedback_list in feedback_by_feature.items():
                avg_rating = sum(f.get('rating', 0) for f in feedback_list) / len(feedback_list)
                
                if avg_rating < 3.0:
                    # This feature needs improvement
                    suggestions.append({
                        'type': 'feature_improvement',
                        'item_id': feature_id,
                        'avg_rating': avg_rating,
                        'feedback_count': len(feedback_list),
                        'suggestion': f"Improve {feature_id} based on user feedback",
                        'priority': 'high' if avg_rating < 2.0 else 'medium'
                    })
        
        # Analyze UI feedback
        if 'ui' in feedback_by_type:
            ui_feedback = feedback_by_type['ui']
            
            # Group by UI element
            feedback_by_element = {}
            for feedback in ui_feedback:
                element_id = feedback.get('item_id')
                if element_id not in feedback_by_element:
                    feedback_by_element[element_id] = []
                feedback_by_element[element_id].append(feedback)
            
            # Find UI elements with low ratings
            for element_id, feedback_list in feedback_by_element.items():
                avg_rating = sum(f.get('rating', 0) for f in feedback_list) / len(feedback_list)
                
                if avg_rating < 3.0:
                    # This UI element needs improvement
                    suggestions.append({
                        'type': 'ui_improvement',
                        'item_id': element_id,
                        'avg_rating': avg_rating,
                        'feedback_count': len(feedback_list),
                        'suggestion': f"Redesign {element_id} based on user feedback",
                        'priority': 'high' if avg_rating < 2.0 else 'medium'
                    })
        
        # Add general improvement suggestions
        suggestions.extend([
            {
                'type': 'new_feature',
                'suggestion': 'Add more advanced AI generation options based on user behavior patterns',
                'priority': 'medium'
            },
            {
                'type': 'performance',
                'suggestion': 'Optimize response time for analytics dashboard based on usage patterns',
                'priority': 'medium'
            },
            {
                'type': 'learning',
                'suggestion': 'Create more intermediate-level tutorials based on user skill progression',
                'priority': 'low'
            }
        ])
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get('priority'), 3))
        
        return suggestions
    
    def _get_all_feedback(self) -> List[Dict[str, Any]]:
        """
        Get all feedback records
        
        Returns:
            List of feedback records
        """
        feedback_list = []
        
        # Traverse user feedback directories
        for user_dir in os.listdir(self.feedback_dir):
            user_feedback_dir = os.path.join(self.feedback_dir, user_dir)
            
            if os.path.isdir(user_feedback_dir) and not user_dir.startswith('items'):
                for filename in os.listdir(user_feedback_dir):
                    if filename.endswith('.json'):
                        feedback_path = os.path.join(user_feedback_dir, filename)
                        
                        try:
                            with open(feedback_path, 'r') as f:
                                feedback = json.load(f)
                                feedback_list.append(feedback)
                        except Exception as e:
                            logger.error(f"Error loading feedback {filename}: {str(e)}")
        
        return feedback_list
    
    def update_behavior_clusters(self) -> None:
        """Update user behavior clusters based on usage patterns"""
        # Get all user profiles
        user_profiles = self._get_all_user_profiles()
        
        if len(user_profiles) < 5:
            logger.info("Not enough users to perform clustering")
            return
        
        # Extract features for clustering
        features = []
        user_ids = []
        
        for profile in user_profiles:
            user_id = profile.get('user_id')
            usage_patterns = profile.get('usage_patterns', {})
            
            # Skip profiles with insufficient data
            if not usage_patterns:
                continue
            
            # Extract feature vector
            feature_vector = []
            
            # Average active hour
            active_hours = usage_patterns.get('active_hours', [])
            avg_hour = sum(active_hours) / len(active_hours) if active_hours else 12
            feature_vector.append(avg_hour / 24)  # Normalize to 0-1
            
            # Session duration
            session_duration = usage_patterns.get('session_duration', 0)
            feature_vector.append(min(1.0, session_duration / 7200))  # Normalize to 0-1, cap at 2 hours
            
            # Feature diversity
            feature_usage = usage_patterns.get('feature_usage', {})
            feature_diversity = len(feature_usage) / 20 if feature_usage else 0  # Normalize to 0-1, assuming max 20 features
            feature_vector.append(feature_diversity)
            
            # Content vs. tool usage ratio
            content_actions = sum(1 for action in usage_patterns.get('frequent_actions', []) if 'content' in action)
            tool_actions = sum(1 for action in usage_patterns.get('frequent_actions', []) if 'tool' in action)
            content_ratio = content_actions / (content_actions + tool_actions) if (content_actions + tool_actions) > 0 else 0.5
            feature_vector.append(content_ratio)
            
            # Add to features list
            features.append(feature_vector)
            user_ids.append(user_id)
        
        if len(features) < 5:
            logger.info("Not enough users with sufficient data to perform clustering")
            return
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Determine optimal number of clusters (simplified)
        max_clusters = min(5, len(features) // 2)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=max_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Update user profiles with cluster assignments
        for i, user_id in enumerate(user_ids):
            cluster = int(clusters[i])
            
            # Update user profile
            profile_path = os.path.join(self.user_models_dir, f"profile_{user_id}.json")
            
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                profile['behavior_cluster'] = cluster
                profile['last_updated'] = int(time.time())
                
                with open(profile_path, 'w') as f:
                    json.dump(profile, f, indent=2)
        
        # Save the cluster model
        import pickle
        with open(os.path.join(self.adaptive_dir, 'behavior_cluster_model.pkl'), 'wb') as f:
            pickle.dump({
                'kmeans': kmeans,
                'scaler': scaler
            }, f)
        
        logger.info(f"Updated behavior clusters for {len(user_ids)} users")
    
    def update_content_relevance_model(self) -> None:
        """Update content relevance prediction model"""
        # Get all feedback
        all_feedback = self._get_all_feedback()
        
        # Filter for content feedback
        content_feedback = [f for f in all_feedback if f.get('item_type') == 'content']
        
        if len(content_feedback) < 20:
            logger.info("Not enough content feedback to train relevance model")
            return
        
        # Prepare training data
        X = []
        y = []
        
        for feedback in content_feedback:
            user_id = feedback.get('user_id')
            content_id = feedback.get('item_id')
            rating = feedback.get('rating', 0)
            
            # Get user profile
            profile = self._get_user_adaptive_profile(user_id)
            
            # Get content metadata (in a real system, you would look this up)
            content_type = content_id.split('_')[0] if '_' in content_id else 'unknown'
            
            # Extract features
            feature_vector = []
            
            # User behavior cluster
            behavior_cluster = profile.get('behavior_cluster')
            for i in range(5):  # Assuming max 5 clusters
                feature_vector.append(1 if behavior_cluster == i else 0)
            
            # User content type preferences
            content_preferences = profile.get('preferences', {}).get('content_types', {})
            preference_score = content_preferences.get(content_type, 0.5)
            feature_vector.append(preference_score)
            
            # User session duration (normalized)
            session_duration = profile.get('usage_patterns', {}).get('session_duration', 0)
            feature_vector.append(min(1.0, session_duration / 7200))  # Normalize to 0-1, cap at 2 hours
            
            # Add to training data
            X.append(feature_vector)
            y.append(rating / 5.0)  # Normalize to 0-1
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Content relevance model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        self.content_relevance_model = model
        
        import pickle
        with open(os.path.join(self.adaptive_dir, 'content_relevance_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    
    def update_feature_usage_model(self) -> None:
        """Update feature usage prediction model"""
        # Get all user profiles
        user_profiles = self._get_all_user_profiles()
        
        # Prepare training data
        X = []
        y = []
        
        for profile in user_profiles:
            usage_patterns = profile.get('usage_patterns', {})
            feature_usage = usage_patterns.get('feature_usage', {})
            
            # Skip profiles with insufficient data
            if not feature_usage:
                continue
            
            # Get behavior cluster
            behavior_cluster = profile.get('behavior_cluster')
            
            # For each feature, create a training example
            for feature_id, usage_score in feature_usage.items():
                # Extract features
                feature_vector = []
                
                # User behavior cluster
                for i in range(5):  # Assuming max 5 clusters
                    feature_vector.append(1 if behavior_cluster == i else 0)
                
                # User session duration (normalized)
                session_duration = usage_patterns.get('session_duration', 0)
                feature_vector.append(min(1.0, session_duration / 7200))  # Normalize to 0-1, cap at 2 hours
                
                # Feature category (in a real system, you would look this up)
                feature_category = feature_id.split('_')[0] if '_' in feature_id else 'unknown'
                categories = ['analytics', 'content', 'collaboration', 'ai', 'security']
                for category in categories:
                    feature_vector.append(1 if feature_category == category else 0)
                
                # Add to training data
                X.append(feature_vector)
                y.append(1 if usage_score > 0.5 else 0)  # Binary classification: frequently used or not
        
        if len(X) < 20:
            logger.info("Not enough feature usage data to train model")
            return
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Feature usage model trained with accuracy: {accuracy:.2f}")
        
        # Save model
        self.feature_usage_model = model
        
        import pickle
        with open(os.path.join(self.adaptive_dir, 'feature_usage_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

# Function to initialize and register the adaptive learning features with the Flask app
def init_adaptive_learning_features(app: Flask, config: Dict[str, Any], analytics_manager=None) -> AdaptiveLearningManager:
    """Initialize and register adaptive learning features with the Flask app"""
    adaptive_learning_manager = AdaptiveLearningManager(app, config, analytics_manager)
    return adaptive_learning_manager
