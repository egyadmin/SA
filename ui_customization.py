"""
Enhanced User Interface Customization Module for Manus Clone

This module provides advanced UI customization capabilities including:
- Theme customization (light/dark modes)
- Layout customization
- UI element personalization
- Accessibility features
- Visual preferences management
"""

import os
import re
import json
import time
import uuid
import logging
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UICustomization:
    """Enhanced User Interface Customization class for UI personalization"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the UI Customization system
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.themes_dir = config.get('THEMES_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'themes'))
        self.layouts_dir = config.get('LAYOUTS_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'layouts'))
        self.user_preferences_file = config.get('USER_PREFERENCES_FILE', os.path.join(os.path.expanduser('~'), '.manus_clone', 'user_preferences.json'))
        
        # Default theme and layout
        self.default_theme = config.get('DEFAULT_THEME', 'light')
        self.default_layout = config.get('DEFAULT_LAYOUT', 'standard')
        
        # Create directories if they don't exist
        os.makedirs(self.themes_dir, exist_ok=True)
        os.makedirs(self.layouts_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.user_preferences_file), exist_ok=True)
        
        # Load user preferences
        self.user_preferences = self._load_user_preferences()
        
        # Initialize built-in themes and layouts
        self._initialize_built_in_themes()
        self._initialize_built_in_layouts()
        
        logger.info(f"UI Customization initialized with theme: {self.get_current_theme_name()} and layout: {self.get_current_layout_name()}")
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """
        Load user preferences from file
        
        Returns:
            Dictionary with user preferences
        """
        try:
            if os.path.exists(self.user_preferences_file):
                with open(self.user_preferences_file, 'r', encoding='utf-8') as f:
                    preferences = json.load(f)
                logger.info(f"Loaded user preferences from {self.user_preferences_file}")
                return preferences
            else:
                # Default preferences
                default_preferences = {
                    'theme': self.default_theme,
                    'layout': self.default_layout,
                    'font_size': 'medium',
                    'animation_enabled': True,
                    'sidebar_collapsed': False,
                    'notifications_enabled': True,
                    'sound_enabled': True,
                    'accessibility': {
                        'high_contrast': False,
                        'screen_reader_optimized': False,
                        'reduced_motion': False,
                        'keyboard_navigation': True
                    },
                    'language': 'en',
                    'time_format': '24h',
                    'date_format': 'YYYY-MM-DD',
                    'custom_css': '',
                    'panels': {
                        'chat': {'visible': True, 'position': 'center', 'size': 'large'},
                        'status': {'visible': True, 'position': 'right', 'size': 'medium'},
                        'tools': {'visible': True, 'position': 'left', 'size': 'medium'},
                        'files': {'visible': True, 'position': 'bottom', 'size': 'medium'}
                    }
                }
                
                # Save default preferences
                self._save_user_preferences(default_preferences)
                logger.info(f"Created default user preferences at {self.user_preferences_file}")
                return default_preferences
        except Exception as e:
            logger.error(f"Error loading user preferences: {str(e)}")
            # Return default preferences without saving
            return {
                'theme': self.default_theme,
                'layout': self.default_layout,
                'font_size': 'medium',
                'animation_enabled': True,
                'sidebar_collapsed': False,
                'notifications_enabled': True,
                'sound_enabled': True,
                'accessibility': {
                    'high_contrast': False,
                    'screen_reader_optimized': False,
                    'reduced_motion': False,
                    'keyboard_navigation': True
                },
                'language': 'en',
                'time_format': '24h',
                'date_format': 'YYYY-MM-DD',
                'custom_css': '',
                'panels': {
                    'chat': {'visible': True, 'position': 'center', 'size': 'large'},
                    'status': {'visible': True, 'position': 'right', 'size': 'medium'},
                    'tools': {'visible': True, 'position': 'left', 'size': 'medium'},
                    'files': {'visible': True, 'position': 'bottom', 'size': 'medium'}
                }
            }
    
    def _save_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences to file
        
        Args:
            preferences: Dictionary with user preferences
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.user_preferences_file, 'w', encoding='utf-8') as f:
                json.dump(preferences, f, indent=2)
            logger.info(f"Saved user preferences to {self.user_preferences_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving user preferences: {str(e)}")
            return False
    
    def _initialize_built_in_themes(self) -> None:
        """
        Initialize built-in themes
        """
        try:
            # Define built-in themes
            built_in_themes = {
                'light': {
                    'name': 'Light',
                    'description': 'Clean light theme with blue accents',
                    'colors': {
                        'primary': '#1976d2',
                        'secondary': '#03a9f4',
                        'background': '#ffffff',
                        'surface': '#f5f5f5',
                        'error': '#b00020',
                        'text': {
                            'primary': '#212121',
                            'secondary': '#757575',
                            'disabled': '#9e9e9e',
                            'hint': '#9e9e9e'
                        },
                        'divider': '#e0e0e0',
                        'shadow': 'rgba(0, 0, 0, 0.1)'
                    },
                    'dark': False,
                    'font': {
                        'family': 'Roboto, Arial, sans-serif',
                        'weights': {
                            'light': 300,
                            'regular': 400,
                            'medium': 500,
                            'bold': 700
                        }
                    },
                    'spacing': {
                        'unit': 8,
                        'xs': 4,
                        'sm': 8,
                        'md': 16,
                        'lg': 24,
                        'xl': 32
                    },
                    'border_radius': {
                        'xs': 2,
                        'sm': 4,
                        'md': 8,
                        'lg': 16,
                        'xl': 24
                    }
                },
                'dark': {
                    'name': 'Dark',
                    'description': 'Modern dark theme with blue accents',
                    'colors': {
                        'primary': '#90caf9',
                        'secondary': '#03dac6',
                        'background': '#121212',
                        'surface': '#1e1e1e',
                        'error': '#cf6679',
                        'text': {
                            'primary': '#ffffff',
                            'secondary': '#b0bec5',
                            'disabled': '#6c7a89',
                            'hint': '#6c7a89'
                        },
                        'divider': '#2d2d2d',
                        'shadow': 'rgba(0, 0, 0, 0.5)'
                    },
                    'dark': True,
                    'font': {
                        'family': 'Roboto, Arial, sans-serif',
                        'weights': {
                            'light': 300,
                            'regular': 400,
                            'medium': 500,
                            'bold': 700
                        }
                    },
                    'spacing': {
                        'unit': 8,
                        'xs': 4,
                        'sm': 8,
                        'md': 16,
                        'lg': 24,
                        'xl': 32
                    },
                    'border_radius': {
                        'xs': 2,
                        'sm': 4,
                        'md': 8,
                        'lg': 16,
                        'xl': 24
                    }
                },
                'high_contrast': {
                    'name': 'High Contrast',
                    'description': 'High contrast theme for accessibility',
                    'colors': {
                        'primary': '#ffff00',
                        'secondary': '#00ffff',
                        'background': '#000000',
                        'surface': '#0a0a0a',
                        'error': '#ff0000',
                        'text': {
                            'primary': '#ffffff',
                            'secondary': '#ffffff',
                            'disabled': '#cccccc',
                            'hint': '#cccccc'
                        },
                        'divider': '#ffffff',
                        'shadow': 'rgba(255, 255, 255, 0.5)'
                    },
                    'dark': True,
                    'font': {
                        'family': 'Arial, sans-serif',
                        'weights': {
                            'light': 400,
                            'regular': 400,
                            'medium': 700,
                            'bold': 700
                        }
                    },
                    'spacing': {
                        'unit': 8,
                        'xs': 4,
                        'sm': 8,
                        'md': 16,
                        'lg': 24,
                        'xl': 32
                    },
                    'border_radius': {
                        'xs': 0,
                        'sm': 0,
                        'md': 0,
                        'lg': 0,
                        'xl': 0
                    }
                },
                'solarized_light': {
                    'name': 'Solarized Light',
                    'description': 'Solarized light theme for reduced eye strain',
                    'colors': {
                        'primary': '#268bd2',
                        'secondary': '#2aa198',
                        'background': '#fdf6e3',
                        'surface': '#eee8d5',
                        'error': '#dc322f',
                        'text': {
                            'primary': '#073642',
                            'secondary': '#586e75',
                            'disabled': '#93a1a1',
                            'hint': '#93a1a1'
                        },
                        'divider': '#eee8d5',
                        'shadow': 'rgba(0, 0, 0, 0.1)'
                    },
                    'dark': False,
                    'font': {
                        'family': 'Roboto, Arial, sans-serif',
                        'weights': {
                            'light': 300,
                            'regular': 400,
                            'medium': 500,
                            'bold': 700
                        }
                    },
                    'spacing': {
                        'unit': 8,
                        'xs': 4,
                        'sm': 8,
                        'md': 16,
                        'lg': 24,
                        'xl': 32
                    },
                    'border_radius': {
                        'xs': 2,
                        'sm': 4,
                        'md': 8,
                        'lg': 16,
                        'xl': 24
                    }
                },
                'solarized_dark': {
                    'name': 'Solarized Dark',
                    'description': 'Solarized dark theme for reduced eye strain',
                    'colors': {
                        'primary': '#268bd2',
                        'secondary': '#2aa198',
                        'background': '#002b36',
                        'surface': '#073642',
                        'error': '#dc322f',
                        'text': {
                            'primary': '#fdf6e3',
                            'secondary': '#eee8d5',
                            'disabled': '#93a1a1',
                            'hint': '#93a1a1'
                        },
                        'divider': '#073642',
                        'shadow': 'rgba(0, 0, 0, 0.5)'
                    },
                    'dark': True,
                    'font': {
                        'family': 'Roboto, Arial, sans-serif',
                        'weights': {
                            'light': 300,
                            'regular': 400,
                            'medium': 500,
                            'bold': 700
                        }
                    },
                    'spacing': {
                        'unit': 8,
                        'xs': 4,
                        'sm': 8,
                        'md': 16,
                        'lg': 24,
                        'xl': 32
                    },
                    'border_radius': {
                        'xs': 2,
                        'sm': 4,
                        'md': 8,
                        'lg': 16,
                        'xl': 24
                    }
                }
            }
            
            # Save built-in themes to files
            for theme_id, theme_data in built_in_themes.items():
                theme_file = os.path.join(self.themes_dir, f"{theme_id}.json")
                if not os.path.exists(theme_file):
                    with open(theme_file, 'w', encoding='utf-8') as f:
                        json.dump(theme_data, f, indent=2)
                    logger.info(f"Created built-in theme: {theme_id}")
            
            logger.info(f"Initialized {len(built_in_themes)} built-in themes")
        except Exception as e:
            logger.error(f"Error initializing built-in themes: {str(e)}")
    
    def _initialize_built_in_layouts(self) -> None:
        """
        Initialize built-in layouts
        """
        try:
            # Define built-in layouts
            built_in_layouts = {
                'standard': {
                    'name': 'Standard',
                    'description': 'Standard layout with sidebar and main content',
                    'structure': {
                        'type': 'horizontal',
                        'children': [
                            {
                     
(Content truncated due to size limit. Use line ranges to read in chunks)