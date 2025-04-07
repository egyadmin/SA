"""
Multilingual Support Module for Manus Clone

This module provides comprehensive multilingual capabilities including:
- Language detection
- Text translation
- Multilingual text processing
- RTL (Right-to-Left) language support
- Language-specific formatting
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

class MultilingualSupport:
    """Multilingual Support class for language processing capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Multilingual Support system
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.default_language = config.get('DEFAULT_LANGUAGE', 'en')
        self.supported_languages = config.get('SUPPORTED_LANGUAGES', {
            'ar': {'name': 'العربية', 'rtl': True, 'enabled': True},
            'en': {'name': 'English', 'rtl': False, 'enabled': True},
            'fr': {'name': 'Français', 'rtl': False, 'enabled': True},
            'es': {'name': 'Español', 'rtl': False, 'enabled': True},
            'de': {'name': 'Deutsch', 'rtl': False, 'enabled': True},
            'zh': {'name': '中文', 'rtl': False, 'enabled': True},
            'ja': {'name': '日本語', 'rtl': False, 'enabled': True},
            'ru': {'name': 'Русский', 'rtl': False, 'enabled': True},
            'hi': {'name': 'हिन्दी', 'rtl': False, 'enabled': True},
            'ur': {'name': 'اردو', 'rtl': True, 'enabled': True},
            'fa': {'name': 'فارسی', 'rtl': True, 'enabled': True},
            'he': {'name': 'עברית', 'rtl': True, 'enabled': True}
        })
        
        # Initialize translation service
        self.translation_service = config.get('TRANSLATION_SERVICE', 'google')
        self.api_keys = config.get('API_KEYS', {})
        
        # Check for required dependencies
        self._check_dependencies()
        
        logger.info(f"Multilingual Support initialized with default language: {self.default_language}")
        logger.info(f"Supported languages: {', '.join(self.supported_languages.keys())}")
    
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are installed
        """
        try:
            # Check Python dependencies
            required_modules = ['langdetect', 'googletrans', 'deep-translator', 'arabic_reshaper', 'python-bidi']
            missing_modules = []
            
            for module in required_modules:
                try:
                    __import__(module.replace('-', '_').split('.')[0])
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                logger.warning(f"Missing Python modules: {', '.join(missing_modules)}. Some features may not work.")
                logger.info(f"Install missing modules with: pip install {' '.join(missing_modules)}")
            else:
                logger.info("All required Python modules are installed")
                
        except Exception as e:
            logger.error(f"Error checking dependencies: {str(e)}")
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a text
        
        Args:
            text: Text to detect language
            
        Returns:
            Dictionary with detected language information
        """
        try:
            from langdetect import detect, detect_langs
            
            if not text or len(text.strip()) < 3:
                return {
                    'language': self.default_language,
                    'confidence': 1.0,
                    'reliable': False,
                    'message': 'Text too short for reliable detection'
                }
            
            # Get primary language
            primary_lang = detect(text)
            
            # Get language probabilities
            lang_probabilities = detect_langs(text)
            
            # Format results
            results = {
                'language': primary_lang,
                'confidence': next((float(lang.prob) for lang in lang_probabilities if lang.lang == primary_lang), 0.0),
                'reliable': len(text.split()) >= 3,  # Consider reliable if at least 3 words
                'alternatives': [{'language': lang.lang, 'confidence': float(lang.prob)} for lang in lang_probabilities if lang.lang != primary_lang]
            }
            
            # Add language name and RTL info if available
            if primary_lang in self.supported_languages:
                results['language_name'] = self.supported_languages[primary_lang]['name']
                results['rtl'] = self.supported_languages[primary_lang]['rtl']
            
            logger.info(f"Detected language: {primary_lang} with confidence {results['confidence']:.2f}")
            return results
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return {
                'language': self.default_language,
                'confidence': 0.0,
                'reliable': False,
                'error': str(e)
            }
    
    def translate_text(self, text: str, target_language: str, source_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (if None, auto-detect)
            
        Returns:
            Dictionary with translation results
        """
        try:
            if not text:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_language or self.default_language,
                    'target_language': target_language
                }
            
            # Auto-detect source language if not provided
            if not source_language:
                detected = self.detect_language(text)
                source_language = detected['language']
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_language,
                    'target_language': target_language,
                    'message': 'Source and target languages are the same, no translation needed'
                }
            
            # Use appropriate translation service
            if self.translation_service == 'google':
                from googletrans import Translator
                translator = Translator()
                result = translator.translate(text, dest=target_language, src=source_language)
                
                translation = {
                    'original_text': text,
                    'translated_text': result.text,
                    'source_language': result.src,
                    'target_language': result.dest,
                    'confidence': getattr(result, 'confidence', None)
                }
            
            elif self.translation_service == 'deepl':
                from deep_translator import DeeplTranslator
                
                # Check if API key is available
                if 'deepl' not in self.api_keys:
                    raise ValueError("DeepL API key not found in configuration")
                
                translator = DeeplTranslator(api_key=self.api_keys['deepl'])
                translated_text = translator.translate(text, source=source_language, target=target_language)
                
                translation = {
                    'original_text': text,
                    'translated_text': translated_text,
                    'source_language': source_language,
                    'target_language': target_language
                }
            
            else:
                # Fallback to a basic translation service
                from deep_translator import GoogleTranslator
                
                translator = GoogleTranslator(source=source_language, target=target_language)
                translated_text = translator.translate(text)
                
                translation = {
                    'original_text': text,
                    'translated_text': translated_text,
                    'source_language': source_language,
                    'target_language': target_language
                }
            
            logger.info(f"Translated text from {source_language} to {target_language}")
            return translation
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language or self.default_language,
                'target_language': target_language,
                'error': str(e)
            }
    
    def get_supported_languages(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of supported languages
        
        Returns:
            Dictionary of supported languages with their properties
        """
        return self.supported_languages
    
    def is_rtl_language(self, language_code: str) -> bool:
        """
        Check if a language is RTL (Right-to-Left)
        
        Args:
            language_code: Language code
            
        Returns:
            True if RTL, False otherwise
        """
        if language_code in self.supported_languages:
            return self.supported_languages[language_code].get('rtl', False)
        return False
    
    def format_rtl_text(self, text: str) -> str:
        """
        Format RTL text for proper display
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text
        """
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            
            # Reshape Arabic text
            reshaped_text = arabic_reshaper.reshape(text)
            
            # Apply bidirectional algorithm
            bidi_text = get_display(reshaped_text)
            
            return bidi_text
        except Exception as e:
            logger.error(f"Error formatting RTL text: {str(e)}")
            return text
    
    def get_language_direction(self, language_code: str) -> str:
        """
        Get text direction for a language
        
        Args:
            language_code: Language code
            
        Returns:
            'rtl' for Right-to-Left languages, 'ltr' for Left-to-Right
        """
        return 'rtl' if self.is_rtl_language(language_code) else 'ltr'
    
    def get_language_font(self, language_code: str) -> Dict[str, Any]:
        """
        Get recommended font settings for a language
        
        Args:
            language_code: Language code
            
        Returns:
            Dictionary with font recommendations
        """
        # Default font settings
        default_fonts = {
            'family': 'Arial, sans-serif',
            'size': '14px',
            'weight': 'normal'
        }
        
        # Language-specific font settings
        language_fonts = {
            'ar': {
                'family': 'Amiri, Scheherazade, Arial, sans-serif',
                'size': '16px',
                'weight': 'normal'
            },
            'zh': {
                'family': 'Noto Sans SC, Microsoft YaHei, SimHei, sans-serif',
                'size': '16px',
                'weight': 'normal'
            },
            'ja': {
                'family': 'Noto Sans JP, Meiryo, sans-serif',
                'size': '16px',
                'weight': 'normal'
            },
            'hi': {
                'family': 'Noto Sans Devanagari, Mangal, sans-serif',
                'size': '16px',
                'weight': 'normal'
            },
            'ur': {
                'family': 'Noto Nastaliq Urdu, Urdu Typesetting, sans-serif',
                'size': '18px',
                'weight': 'normal'
            },
            'fa': {
                'family': 'Vazir, Tahoma, sans-serif',
                'size': '16px',
                'weight': 'normal'
            },
            'he': {
                'family': 'Noto Sans Hebrew, Arial Hebrew, sans-serif',
                'size': '16px',
                'weight': 'normal'
            }
        }
        
        return language_fonts.get(language_code, default_fonts)
    
    def get_language_name(self, language_code: str, in_native_language: bool = False) -> str:
        """
        Get the name of a language
        
        Args:
            language_code: Language code
            in_native_language: Whether to return the name in the native language
            
        Returns:
            Language name
        """
        if language_code in self.supported_languages:
            if in_native_language:
                return self.supported_languages[language_code]['name']
            else:
                # English names of languages
                english_names = {
                    'ar': 'Arabic',
                    'en': 'English',
                    'fr': 'French',
                    'es': 'Spanish',
                    'de': 'German',
                    'zh': 'Chinese',
                    'ja': 'Japanese',
                    'ru': 'Russian',
                    'hi': 'Hindi',
                    'ur': 'Urdu',
                    'fa': 'Persian',
                    'he': 'Hebrew'
                }
                return english_names.get(language_code, language_code)
        return language_code
    
    def translate_batch(self, texts: List[str], target_language: str, source_language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Translate a batch of texts
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (if None, auto-detect)
            
        Returns:
            List of dictionaries with translation results
        """
        results = []
        for text in texts:
            result = self.translate_text(text, target_language, source_language)
            results.append(result)
        return results
    
    def detect_language_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect language for a batch of texts
        
        Args:
            texts: List of texts to detect language
            
        Returns:
            List of dictionaries with detected language information
        """
        results = []
        for text in texts:
            result = self.detect_language(text)
            results.append(result)
        return results
    
    def get_language_keyboard_layout(self, language_code: str) -> Dict[str, Any]:
        """
        Get keyboard layout information for a language
        
        Args:
            language_code: Language code
            
        Returns:
            Dictionary with keyboard layout information
        """
        # Default keyboard layout
        default_layout = {
            'layout': 'qwerty',
            'input_method': 'latin',
            'special_keys': []
        }
        
        # Language-specific keyboard layouts
        language_layouts = {
            'ar': {
                'layout': 'arabic',
                'input_method': 'arabic'
(Content truncated due to size limit. Use line ranges to read in chunks)