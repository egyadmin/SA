"""
IoT Integration Module for Manus Clone

This module implements IoT device integration features including:
- IoT device API interfaces
- Support for common IoT protocols
- Device management system
- Integration with analytics and adaptive learning systems
"""

import os
import json
import time
import logging
import uuid
import datetime
import threading
import queue
import paho.mqtt.client as mqtt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IoTManager:
    """Manages IoT device integration for the Manus Clone application"""
    
    def __init__(self, app: Flask, config: Dict[str, Any], analytics_manager=None, adaptive_learning_manager=None):
        """
        Initialize the IoT Manager with configuration
        
        Args:
            app: Flask application instance
            config: Dictionary containing configuration parameters
            analytics_manager: Optional reference to the Analytics Manager for data integration
            adaptive_learning_manager: Optional reference to the Adaptive Learning Manager for behavior adaptation
        """
        self.config = config
        self.app = app
        self.analytics_manager = analytics_manager
        self.adaptive_learning_manager = adaptive_learning_manager
        
        # Initialize storage directories
        self.iot_dir = os.path.join(config.get('data_dir', 'data'), 'iot')
        os.makedirs(self.iot_dir, exist_ok=True)
        
        # Devices storage
        self.devices_dir = os.path.join(self.iot_dir, 'devices')
        os.makedirs(self.devices_dir, exist_ok=True)
        
        # Data storage
        self.data_dir = os.path.join(self.iot_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize protocol handlers
        self._init_protocol_handlers()
        
        # Initialize device registry
        self.device_registry = {}
        self._load_device_registry()
        
        # Initialize message queues for device communication
        self.message_queues = {}
        
        # Initialize device type handlers
        self.device_type_handlers = {}
        self._init_device_type_handlers()
        
        # Register routes
        self._register_routes()
        
        logger.info("IoT Manager initialized successfully")
    
    def _init_protocol_handlers(self):
        """Initialize protocol handlers for different IoT protocols"""
        # MQTT protocol handler
        self.mqtt_client = None
        self.mqtt_connected = False
        self.mqtt_topics = {}
        
        # CoAP protocol handler (simplified)
        self.coap_server = None
        self.coap_resources = {}
        
        # HTTP/REST protocol handler
        self.http_endpoints = {}
        
        # Initialize protocol configurations
        mqtt_config = self.config.get('mqtt', {})
        self.mqtt_broker = mqtt_config.get('broker', 'localhost')
        self.mqtt_port = mqtt_config.get('port', 1883)
        self.mqtt_keepalive = mqtt_config.get('keepalive', 60)
        self.mqtt_username = mqtt_config.get('username')
        self.mqtt_password = mqtt_config.get('password')
        
        # Start protocol handlers if configured
        if self.config.get('enable_mqtt', True):
            self._start_mqtt_client()
        
        if self.config.get('enable_coap', False):
            self._start_coap_server()
    
    def _start_mqtt_client(self):
        """Start MQTT client for device communication"""
        try:
            # Create MQTT client
            client_id = f"manus-clone-{uuid.uuid4().hex[:8]}"
            self.mqtt_client = mqtt.Client(client_id=client_id)
            
            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Set authentication if provided
            if self.mqtt_username and self.mqtt_password:
                self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)
            
            # Connect to broker
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, self.mqtt_keepalive)
            
            # Start the loop in a separate thread
            self.mqtt_client.loop_start()
            
            logger.info(f"MQTT client started, connecting to {self.mqtt_broker}:{self.mqtt_port}")
        
        except Exception as e:
            logger.error(f"Error starting MQTT client: {str(e)}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for when the MQTT client connects to the broker"""
        if rc == 0:
            self.mqtt_connected = True
            logger.info("Connected to MQTT broker")
            
            # Subscribe to device discovery topic
            discovery_topic = "manus/discovery/#"
            client.subscribe(discovery_topic)
            logger.info(f"Subscribed to {discovery_topic}")
            
            # Subscribe to registered device topics
            for device_id, device in self.device_registry.items():
                if device.get('protocol') == 'mqtt':
                    topic = device.get('topic')
                    if topic:
                        client.subscribe(topic)
                        logger.info(f"Subscribed to {topic} for device {device_id}")
        else:
            self.mqtt_connected = False
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Callback for when a message is received from the MQTT broker"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.debug(f"Received MQTT message on topic {topic}: {payload}")
            
            # Handle discovery messages
            if topic.startswith("manus/discovery/"):
                self._handle_device_discovery(topic, payload)
                return
            
            # Handle device messages
            for device_id, device in self.device_registry.items():
                if device.get('protocol') == 'mqtt' and device.get('topic') == topic:
                    self._handle_device_message(device_id, payload)
                    return
            
            # Unknown topic
            logger.warning(f"Received message on unknown topic: {topic}")
        
        except Exception as e:
            logger.error(f"Error handling MQTT message: {str(e)}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for when the MQTT client disconnects from the broker"""
        self.mqtt_connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection, return code: {rc}")
            # Try to reconnect
            try:
                client.reconnect()
            except Exception as e:
                logger.error(f"Error reconnecting to MQTT broker: {str(e)}")
        else:
            logger.info("MQTT client disconnected")
    
    def _start_coap_server(self):
        """Start CoAP server for device communication"""
        # This is a placeholder - in a real implementation, you would use a CoAP library
        logger.info("CoAP server functionality is not fully implemented")
    
    def _load_device_registry(self):
        """Load device registry from storage"""
        registry_path = os.path.join(self.iot_dir, 'device_registry.json')
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.device_registry = json.load(f)
                logger.info(f"Loaded {len(self.device_registry)} devices from registry")
            except Exception as e:
                logger.error(f"Error loading device registry: {str(e)}")
                self.device_registry = {}
    
    def _save_device_registry(self):
        """Save device registry to storage"""
        registry_path = os.path.join(self.iot_dir, 'device_registry.json')
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.device_registry, f, indent=2)
            logger.info(f"Saved {len(self.device_registry)} devices to registry")
        except Exception as e:
            logger.error(f"Error saving device registry: {str(e)}")
    
    def _init_device_type_handlers(self):
        """Initialize handlers for different device types"""
        # Register handlers for common device types
        self.device_type_handlers = {
            'sensor': self._handle_sensor_data,
            'actuator': self._handle_actuator_data,
            'camera': self._handle_camera_data,
            'display': self._handle_display_data,
            'switch': self._handle_switch_data,
            'thermostat': self._handle_thermostat_data,
            'light': self._handle_light_data,
            'lock': self._handle_lock_data,
            'speaker': self._handle_speaker_data,
            'generic': self._handle_generic_data
        }
    
    def _register_routes(self):
        """Register HTTP routes for IoT features"""
        
        @self.app.route('/api/iot/devices', methods=['GET'])
        def get_devices():
            """Get all registered devices"""
            return jsonify({
                'success': True,
                'devices': list(self.device_registry.values())
            })
        
        @self.app.route('/api/iot/devices/<device_id>', methods=['GET'])
        def get_device(device_id):
            """Get a specific device"""
            if device_id in self.device_registry:
                return jsonify({
                    'success': True,
                    'device': self.device_registry[device_id]
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} not found"
                }), 404
        
        @self.app.route('/api/iot/devices', methods=['POST'])
        def register_device():
            """Register a new device"""
            data = request.get_json()
            
            if 'name' not in data or 'type' not in data or 'protocol' not in data:
                return jsonify({
                    'success': False,
                    'error': "Name, type, and protocol are required"
                }), 400
            
            # Create device ID if not provided
            device_id = data.get('id', str(uuid.uuid4()))
            
            # Check if device already exists
            if device_id in self.device_registry:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} already exists"
                }), 409
            
            # Create device record
            device = {
                'id': device_id,
                'name': data['name'],
                'type': data['type'],
                'protocol': data['protocol'],
                'status': 'registered',
                'registered_at': int(time.time()),
                'last_seen': None,
                'properties': data.get('properties', {}),
                'metadata': data.get('metadata', {})
            }
            
            # Add protocol-specific properties
            if data['protocol'] == 'mqtt':
                device['topic'] = data.get('topic', f"manus/devices/{device_id}")
                
                # Subscribe to device topic if MQTT is enabled
                if self.mqtt_client and self.mqtt_connected:
                    self.mqtt_client.subscribe(device['topic'])
            
            elif data['protocol'] == 'coap':
                device['resource_uri'] = data.get('resource_uri', f"/devices/{device_id}")
            
            elif data['protocol'] == 'http':
                device['endpoint'] = data.get('endpoint')
            
            # Add device to registry
            self.device_registry[device_id] = device
            self._save_device_registry()
            
            # Create device directory
            device_dir = os.path.join(self.devices_dir, device_id)
            os.makedirs(device_dir, exist_ok=True)
            
            return jsonify({
                'success': True,
                'device': device
            })
        
        @self.app.route('/api/iot/devices/<device_id>', methods=['PUT'])
        def update_device(device_id):
            """Update a device"""
            if device_id not in self.device_registry:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} not found"
                }), 404
            
            data = request.get_json()
            device = self.device_registry[device_id]
            
            # Update device properties
            if 'name' in data:
                device['name'] = data['name']
            
            if 'properties' in data:
                device['properties'].update(data['properties'])
            
            if 'metadata' in data:
                device['metadata'].update(data['metadata'])
            
            # Update protocol-specific properties
            if device['protocol'] == 'mqtt' and 'topic' in data:
                old_topic = device.get('topic')
                new_topic = data['topic']
                
                if old_topic != new_topic and self.mqtt_client and self.mqtt_connected:
                    # Unsubscribe from old topic
                    if old_topic:
                        self.mqtt_client.unsubscribe(old_topic)
                    
                    # Subscribe to new topic
                    self.mqtt_client.subscribe(new_topic)
                    device['topic'] = new_topic
            
            # Save updated registry
            self._save_device_registry()
            
            return jsonify({
                'success': True,
                'device': device
            })
        
        @self.app.route('/api/iot/devices/<device_id>', methods=['DELETE'])
        def delete_device(device_id):
            """Delete a device"""
            if device_id not in self.device_registry:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} not found"
                }), 404
            
            device = self.device_registry[device_id]
            
            # Unsubscribe from MQTT topic if applicable
            if device['protocol'] == 'mqtt' and 'topic' in device and self.mqtt_client and self.mqtt_connected:
                self.mqtt_client.unsubscribe(device['topic'])
            
            # Remove device from registry
            del self.device_registry[device_id]
            self._save_device_registry()
            
            return jsonify({
                'success': True,
                'message': f"Device {device_id} deleted successfully"
            })
        
        @self.app.route('/api/iot/devices/<device_id>/command', methods=['POST'])
        def send_device_command(device_id):
            """Send a command to a device"""
            if device_id not in self.device_registry:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} not found"
                }), 404
            
            data = request.get_json()
            
            if 'command' not in data:
                return jsonify({
                    'success': False,
                    'error': "Command is required"
                }), 400
            
            command = data['command']
            params = data.get('params', {})
            
            # Send command to device
            success, result = self._send_device_command(device_id, command, params)
            
            if success:
                return jsonify({
                    'success': True,
                    'result': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result
                }), 500
        
        @self.app.route('/api/iot/devices/<device_id>/data', methods=['GET'])
        def get_device_data(device_id):
            """Get data from a device"""
            if device_id not in self.device_registry:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} not found"
                }), 404
            
            # Get parameters
            start_time = request.args.get('start_time')
            end_time = request.args.get('end_time')
            limit = request.args.get('limit', 100)
            
            try:
                limit = int(limit)
            except ValueError:
                limit = 100
            
            # Convert time strings to timestamps if provided
            if start_time:
                try:
                    start_time = int(datetime.datetime.fromisoformat(start_time).timestamp())
                except ValueError:
                    try:
                        start_time = int(start_time)
                    except ValueError:
                        start_time = None
            
            if end_time:
                try:
                    end_time = int(datetime.datetime.fromisoformat(end_time).timestamp())
                except ValueError:
                    try:
                        end_time = int(end_time)
                    except ValueError:
                        end_time = None
            
            # Get device data
            data = self._get_device_data(device_id, start_time, end_time, limit)
            
            return jsonify({
                'success': True,
                'device_id': device_id,
                'data': data
            })
        
        @self.app.route('/api/iot/devices/<device_id>/data', methods=['POST'])
        def submit_device_data(device_id):
            """Submit data for a device (for HTTP devices)"""
            if device_id not in self.device_registry:
                return jsonify({
                    'success': False,
                    'error': f"Device {device_id} not found"
                }), 404
            
            data = request.get_json()
            
            if 'data' not in data:
                return jsonify({
                    'success': False,
                    'error': "Data is required"
                }), 400
            
            # Process device data
            self._handle_device_message(device_id, json.dumps(data['data']))
            
            return jsonify({
                'success': True,
                'message': "Data received successfully"
            })
        
        @self.app.route('/api/iot/discovery', methods=['POST'])
        def start_device_discovery():
            """Start device discovery"""
            data = request.get_json()
            
            protocol = data.get('protocol', 'mqtt')
            duration = data.get('duration', 60)  # seconds
            
            # Start discovery process
            discovery_id = self._start_discovery(protocol, duration)
            
            return jsonify({
                'success': True,
                'discovery_id': discovery_id,
                'message': f"Discovery started for {protocol} protocol, will run for {duration} seconds"
            })
        
        @self.app.route('/api/iot/discovery/<discovery_id>', methods=['GET'])
        def get_discovery_results(discovery_id):
            """Get results of a discovery process"""
            # This is a placeholder - in a real implementation, you would track discovery processes
            return jsonify({
                'success': True,
                'discovery_id': discovery_id,
                'status': 'completed',
                'devices_found': []
            })
        
        @self.app.route('/api/iot/dashboard', methods=['GET'])
        def get_iot_dashboard():
            """Get IoT dashboard data"""
            # Get device statistics
            total_devices = len(self.device_registry)
            devices_by_type = {}
            devices_by_protocol = {}
            active_devices = 0
            
            for device in self.device_registry.values():
                # Count by type
                device_type = device.get('type', 'unknown')
                if device_type not in devices_by_type:
                    devices_by_type[device_type] = 0
                devices_by_type[device_type] += 1
                
                # Count by protocol
                protocol = device.get('protocol', 'unknown')
                if protocol not in devices_by_protocol:
                    devices_by_protocol[protocol] = 0
                devices_by_protocol[protocol] += 1
                
                # Count active devices
                last_seen = device.get('last_seen')
                if last_seen and (int(time.time()) - last_seen) < 3600:  # Active in the last hour
                    active_devices += 1
            
            # Get recent data points
            recent_data = self._get_recent_data(10)
            
            # Compile dashboard data
            dashboard_data = {
                'total_devices': total_devices,
                'active_devices': active_devices,
                'devices_by_type': devices_by_type,
                'devices_by_protocol': devices_by_protocol,
                'recent_data': recent_data
            }
            
            return jsonify({
                'success': True,
                'dashboard': dashboard_data
            })
    
    def _handle_device_discovery(self, topic: str, payload: str) -> None:
        """
        Handle device discovery message
        
        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            # Parse discovery payload
            discovery_data = json.loads(payload)
            
            # Extract device information
            device_id = discovery_data.get('id', str(uuid.uuid4()))
            device_name = discovery_data.get('name', f"Device {device_id}")
            device_type = discovery_data.get('type', 'generic')
            
            # Check if device already exists
            if device_id in self.device_registry:
                logger.info(f"Discovered existing device: {device_id}")
                
                # Update last seen timestamp
                self.device_registry[device_id]['last_seen'] = int(time.time())
                self._save_device_registry()
                return
            
            # Create new device record
            device = {
                'id': device_id,
                'name': device_name,
                'type': device_type,
                'protocol': 'mqtt',
                'topic': discovery_data.get('topic', f"manus/devices/{device_id}"),
                'status': 'discovered',
                'registered_at': int(time.time()),
                'last_seen': int(time.time()),
                'properties': discovery_data.get('properties', {}),
                'metadata': discovery_data.get('metadata', {})
            }
            
            # Add device to registry
            self.device_registry[device_id] = device
            self._save_device_registry()
            
            # Create device directory
            device_dir = os.path.join(self.devices_dir, device_id)
            os.makedirs(device_dir, exist_ok=True)
            
            # Subscribe to device topic
            if self.mqtt_client and self.mqtt_connected:
                self.mqtt_client.subscribe(device['topic'])
            
            logger.info(f"Discovered new device: {device_id} ({device_name})")
        
        except Exception as e:
            logger.error(f"Error handling discovery message: {str(e)}")
    
    def _handle_device_message(self, device_id: str, payload: str) -> None:
        """
        Handle message from a device
        
        Args:
            device_id: Device ID
            payload: Message payload
        """
        try:
            # Parse message payload
            message_data = json.loads(payload)
            
            # Update device last seen timestamp
            if device_id in self.device_registry:
                self.device_registry[device_id]['last_seen'] = int(time.time())
                self._save_device_registry()
            
            # Store device data
            self._store_device_data(device_id, message_data)
            
            # Process data based on device type
            if device_id in self.device_registry:
                device_type = self.device_registry[device_id].get('type', 'generic')
                
                if device_type in self.device_type_handlers:
                    self.device_type_handlers[device_type](device_id, message_data)
                else:
                    self._handle_generic_data(device_id, message_data)
            
            # Send data to analytics manager if available
            if self.analytics_manager:
                self._send_to_analytics(device_id, message_data)
            
            # Send data to adaptive learning manager if available
            if self.adaptive_learning_manager:
                self._send_to_adaptive_learning(device_id, message_data)
            
            logger.debug(f"Processed message from device {device_id}")
        
        except Exception as e:
            logger.error(f"Error handling device message: {str(e)}")
    
    def _store_device_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Store data from a device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Create timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = int(time.time())
        
        # Create data directory for device
        device_data_dir = os.path.join(self.data_dir, device_id)
        os.makedirs(device_data_dir, exist_ok=True)
        
        # Create date-based directory
        date_str = datetime.datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d')
        date_dir = os.path.join(device_data_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Generate filename
        filename = f"{data['timestamp']}_{uuid.uuid4().hex[:8]}.json"
        file_path = os.path.join(date_dir, filename)
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_device_data(self, device_id: str, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get data from a device
        
        Args:
            device_id: Device ID
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            limit: Maximum number of data points to return
            
        Returns:
            List of data points
        """
        device_data_dir = os.path.join(self.data_dir, device_id)
        
        if not os.path.exists(device_data_dir):
            return []
        
        # Get all data files for the device
        data_files = []
        
        # If time range is specified, only look in relevant date directories
        if start_time or end_time:
            if start_time:
                start_date = datetime.datetime.fromtimestamp(start_time).date()
            else:
                # Default to 7 days ago
                start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).date()
            
            if end_time:
                end_date = datetime.datetime.fromtimestamp(end_time).date()
            else:
                end_date = datetime.datetime.now().date()
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                date_dir = os.path.join(device_data_dir, date_str)
                
                if os.path.exists(date_dir):
                    for filename in os.listdir(date_dir):
                        if filename.endswith('.json'):
                            file_path = os.path.join(date_dir, filename)
                            data_files.append(file_path)
                
                current_date += datetime.timedelta(days=1)
        else:
            # Get all data files
            for date_dir in os.listdir(device_data_dir):
                date_path = os.path.join(device_data_dir, date_dir)
                
                if os.path.isdir(date_path):
                    for filename in os.listdir(date_path):
                        if filename.endswith('.json'):
                            file_path = os.path.join(date_path, filename)
                            data_files.append(file_path)
        
        # Sort files by timestamp (extracted from filename)
        data_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        # Load data from files
        data_points = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Filter by timestamp if specified
                timestamp = data.get('timestamp', 0)
                
                if start_time and timestamp < start_time:
                    continue
                
                if end_time and timestamp > end_time:
                    continue
                
                data_points.append(data)
                
                # Limit number of data points
                if len(data_points) >= limit:
                    break
            
            except Exception as e:
                logger.error(f"Error loading data file {file_path}: {str(e)}")
        
        return data_points
    
    def _get_recent_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent data from all devices
        
        Args:
            limit: Maximum number of data points to return
            
        Returns:
            List of recent data points
        """
        # Get all data files
        data_files = []
        
        for device_id in os.listdir(self.data_dir):
            device_data_dir = os.path.join(self.data_dir, device_id)
            
            if os.path.isdir(device_data_dir):
                # Get most recent date directory
                date_dirs = [d for d in os.listdir(device_data_dir) if os.path.isdir(os.path.join(device_data_dir, d))]
                date_dirs.sort(reverse=True)
                
                if date_dirs:
                    recent_date_dir = os.path.join(device_data_dir, date_dirs[0])
                    
                    # Get data files in this directory
                    for filename in os.listdir(recent_date_dir):
                        if filename.endswith('.json'):
                            file_path = os.path.join(recent_date_dir, filename)
                            data_files.append((file_path, device_id))
        
        # Sort files by timestamp (extracted from filename)
        data_files.sort(key=lambda x: int(os.path.basename(x[0]).split('_')[0]), reverse=True)
        
        # Load data from files
        recent_data = []
        
        for file_path, device_id in data_files[:limit]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Add device ID to data
                data['device_id'] = device_id
                
                # Add device name if available
                if device_id in self.device_registry:
                    data['device_name'] = self.device_registry[device_id].get('name', device_id)
                
                recent_data.append(data)
            
            except Exception as e:
                logger.error(f"Error loading data file {file_path}: {str(e)}")
        
        return recent_data
    
    def _send_device_command(self, device_id: str, command: str, params: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Send a command to a device
        
        Args:
            device_id: Device ID
            command: Command name
            params: Command parameters
            
        Returns:
            Tuple of (success flag, result/error message)
        """
        if device_id not in self.device_registry:
            return False, f"Device {device_id} not found"
        
        device = self.device_registry[device_id]
        protocol = device.get('protocol')
        
        # Create command message
        command_message = {
            'command': command,
            'params': params,
            'id': str(uuid.uuid4()),
            'timestamp': int(time.time())
        }
        
        # Send command based on protocol
        if protocol == 'mqtt':
            return self._send_mqtt_command(device, command_message)
        elif protocol == 'coap':
            return self._send_coap_command(device, command_message)
        elif protocol == 'http':
            return self._send_http_command(device, command_message)
        else:
            return False, f"Unsupported protocol: {protocol}"
    
    def _send_mqtt_command(self, device: Dict[str, Any], command_message: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Send a command to a device via MQTT
        
        Args:
            device: Device record
            command_message: Command message
            
        Returns:
            Tuple of (success flag, result/error message)
        """
        if not self.mqtt_client or not self.mqtt_connected:
            return False, "MQTT client not connected"
        
        device_id = device['id']
        
        # Get command topic
        command_topic = device.get('command_topic')
        if not command_topic:
            # Use default command topic
            command_topic = f"manus/devices/{device_id}/commands"
        
        try:
            # Convert command message to JSON
            payload = json.dumps(command_message)
            
            # Publish command
            result = self.mqtt_client.publish(command_topic, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Sent command to device {device_id}: {command_message['command']}")
                return True, "Command sent successfully"
            else:
                logger.error(f"Error sending command to device {device_id}: {result.rc}")
                return False, f"MQTT publish error: {result.rc}"
        
        except Exception as e:
            logger.error(f"Error sending MQTT command: {str(e)}")
            return False, str(e)
    
    def _send_coap_command(self, device: Dict[str, Any], command_message: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Send a command to a device via CoAP
        
        Args:
            device: Device record
            command_message: Command message
            
        Returns:
            Tuple of (success flag, result/error message)
        """
        # This is a placeholder - in a real implementation, you would use a CoAP library
        logger.warning("CoAP command functionality is not fully implemented")
        return False, "CoAP commands not implemented"
    
    def _send_http_command(self, device: Dict[str, Any], command_message: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Send a command to a device via HTTP
        
        Args:
            device: Device record
            command_message: Command message
            
        Returns:
            Tuple of (success flag, result/error message)
        """
        endpoint = device.get('endpoint')
        
        if not endpoint:
            return False, "Device endpoint not specified"
        
        try:
            import requests
            
            # Send HTTP POST request
            response = requests.post(
                endpoint,
                json=command_message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Sent command to device {device['id']}: {command_message['command']}")
                
                try:
                    return True, response.json()
                except ValueError:
                    return True, response.text
            else:
                logger.error(f"Error sending command to device {device['id']}: {response.status_code}")
                return False, f"HTTP error: {response.status_code} - {response.text}"
        
        except Exception as e:
            logger.error(f"Error sending HTTP command: {str(e)}")
            return False, str(e)
    
    def _start_discovery(self, protocol: str, duration: int) -> str:
        """
        Start device discovery
        
        Args:
            protocol: Protocol to use for discovery
            duration: Discovery duration in seconds
            
        Returns:
            Discovery ID
        """
        discovery_id = str(uuid.uuid4())
        
        if protocol == 'mqtt':
            self._start_mqtt_discovery(discovery_id, duration)
        elif protocol == 'coap':
            self._start_coap_discovery(discovery_id, duration)
        else:
            logger.warning(f"Discovery not implemented for protocol: {protocol}")
        
        return discovery_id
    
    def _start_mqtt_discovery(self, discovery_id: str, duration: int) -> None:
        """
        Start MQTT device discovery
        
        Args:
            discovery_id: Discovery ID
            duration: Discovery duration in seconds
        """
        if not self.mqtt_client or not self.mqtt_connected:
            logger.error("MQTT client not connected, cannot start discovery")
            return
        
        try:
            # Publish discovery request
            discovery_request = {
                'discovery_id': discovery_id,
                'timestamp': int(time.time()),
                'duration': duration
            }
            
            self.mqtt_client.publish('manus/discovery', json.dumps(discovery_request))
            logger.info(f"Started MQTT discovery {discovery_id} for {duration} seconds")
            
            # Schedule end of discovery
            threading.Timer(duration, self._end_mqtt_discovery, args=[discovery_id]).start()
        
        except Exception as e:
            logger.error(f"Error starting MQTT discovery: {str(e)}")
    
    def _end_mqtt_discovery(self, discovery_id: str) -> None:
        """
        End MQTT device discovery
        
        Args:
            discovery_id: Discovery ID
        """
        logger.info(f"Ended MQTT discovery {discovery_id}")
    
    def _start_coap_discovery(self, discovery_id: str, duration: int) -> None:
        """
        Start CoAP device discovery
        
        Args:
            discovery_id: Discovery ID
            duration: Discovery duration in seconds
        """
        # This is a placeholder - in a real implementation, you would use a CoAP library
        logger.warning("CoAP discovery functionality is not fully implemented")
    
    def _send_to_analytics(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Send device data to analytics manager
        
        Args:
            device_id: Device ID
            data: Device data
        """
        if not self.analytics_manager:
            return
        
        try:
            # Create analytics event
            event_type = 'iot_device_data'
            event_data = {
                'device_id': device_id,
                'device_type': self.device_registry.get(device_id, {}).get('type', 'unknown'),
                'timestamp': data.get('timestamp', int(time.time())),
                'data': data
            }
            
            # Track event using analytics manager
            if hasattr(self.analytics_manager, '_track_event'):
                self.analytics_manager._track_event(event_type, event_data)
        
        except Exception as e:
            logger.error(f"Error sending data to analytics: {str(e)}")
    
    def _send_to_adaptive_learning(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Send device data to adaptive learning manager
        
        Args:
            device_id: Device ID
            data: Device data
        """
        if not self.adaptive_learning_manager:
            return
        
        try:
            # Create interaction data
            interaction_type = 'iot_device_interaction'
            interaction_data = {
                'device_id': device_id,
                'device_type': self.device_registry.get(device_id, {}).get('type', 'unknown'),
                'timestamp': data.get('timestamp', int(time.time())),
                'data': data
            }
            
            # Track interaction using adaptive learning manager
            if hasattr(self.adaptive_learning_manager, '_track_interaction'):
                # Use a generic user ID for device interactions
                self.adaptive_learning_manager._track_interaction('system', interaction_type, interaction_data)
        
        except Exception as e:
            logger.error(f"Error sending data to adaptive learning: {str(e)}")
    
    def _handle_sensor_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a sensor device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract sensor readings
        readings = data.get('readings', {})
        
        if readings:
            logger.debug(f"Sensor {device_id} readings: {readings}")
            
            # Check for alerts or anomalies
            self._check_sensor_alerts(device_id, readings)
    
    def _check_sensor_alerts(self, device_id: str, readings: Dict[str, Any]) -> None:
        """
        Check sensor readings for alerts or anomalies
        
        Args:
            device_id: Device ID
            readings: Sensor readings
        """
        if device_id not in self.device_registry:
            return
        
        device = self.device_registry[device_id]
        
        # Get alert thresholds from device properties
        thresholds = device.get('properties', {}).get('alert_thresholds', {})
        
        if not thresholds:
            return
        
        # Check each reading against thresholds
        alerts = []
        
        for key, value in readings.items():
            if key in thresholds:
                threshold = thresholds[key]
                
                if 'min' in threshold and value < threshold['min']:
                    alerts.append({
                        'type': 'below_threshold',
                        'sensor': key,
                        'value': value,
                        'threshold': threshold['min'],
                        'timestamp': int(time.time())
                    })
                
                if 'max' in threshold and value > threshold['max']:
                    alerts.append({
                        'type': 'above_threshold',
                        'sensor': key,
                        'value': value,
                        'threshold': threshold['max'],
                        'timestamp': int(time.time())
                    })
        
        # Handle alerts
        if alerts:
            logger.warning(f"Sensor {device_id} alerts: {alerts}")
            
            # Store alerts
            self._store_device_alerts(device_id, alerts)
    
    def _store_device_alerts(self, device_id: str, alerts: List[Dict[str, Any]]) -> None:
        """
        Store device alerts
        
        Args:
            device_id: Device ID
            alerts: List of alerts
        """
        # Create alerts directory for device
        device_dir = os.path.join(self.devices_dir, device_id)
        alerts_dir = os.path.join(device_dir, 'alerts')
        os.makedirs(alerts_dir, exist_ok=True)
        
        # Create alert record
        alert_record = {
            'device_id': device_id,
            'timestamp': int(time.time()),
            'alerts': alerts
        }
        
        # Generate filename
        filename = f"{alert_record['timestamp']}_{uuid.uuid4().hex[:8]}.json"
        file_path = os.path.join(alerts_dir, filename)
        
        # Save alerts to file
        with open(file_path, 'w') as f:
            json.dump(alert_record, f, indent=2)
    
    def _handle_actuator_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from an actuator device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract actuator state
        state = data.get('state', {})
        
        if state:
            logger.debug(f"Actuator {device_id} state: {state}")
            
            # Update device properties with current state
            if device_id in self.device_registry:
                self.device_registry[device_id].setdefault('properties', {})['state'] = state
                self._save_device_registry()
    
    def _handle_camera_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a camera device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Check for image data
        if 'image' in data:
            # In a real implementation, you would handle image data (e.g., save to file, process, etc.)
            logger.debug(f"Camera {device_id} image received")
        
        # Check for video data
        if 'video' in data:
            # In a real implementation, you would handle video data
            logger.debug(f"Camera {device_id} video received")
        
        # Check for motion detection
        if 'motion_detected' in data and data['motion_detected']:
            logger.info(f"Camera {device_id} motion detected")
            
            # In a real implementation, you might trigger an alert or other action
    
    def _handle_display_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a display device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract display state
        state = data.get('state', {})
        
        if state:
            logger.debug(f"Display {device_id} state: {state}")
            
            # Update device properties with current state
            if device_id in self.device_registry:
                self.device_registry[device_id].setdefault('properties', {})['state'] = state
                self._save_device_registry()
    
    def _handle_switch_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a switch device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract switch state
        state = data.get('state')
        
        if state is not None:
            logger.debug(f"Switch {device_id} state: {state}")
            
            # Update device properties with current state
            if device_id in self.device_registry:
                self.device_registry[device_id].setdefault('properties', {})['state'] = state
                self._save_device_registry()
    
    def _handle_thermostat_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a thermostat device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract thermostat data
        temperature = data.get('temperature')
        target = data.get('target')
        mode = data.get('mode')
        
        if temperature is not None or target is not None or mode is not None:
            logger.debug(f"Thermostat {device_id} data: temp={temperature}, target={target}, mode={mode}")
            
            # Update device properties
            if device_id in self.device_registry:
                properties = self.device_registry[device_id].setdefault('properties', {})
                
                if temperature is not None:
                    properties['temperature'] = temperature
                
                if target is not None:
                    properties['target'] = target
                
                if mode is not None:
                    properties['mode'] = mode
                
                self._save_device_registry()
    
    def _handle_light_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a light device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract light state
        state = data.get('state')
        brightness = data.get('brightness')
        color = data.get('color')
        
        if state is not None or brightness is not None or color is not None:
            logger.debug(f"Light {device_id} data: state={state}, brightness={brightness}, color={color}")
            
            # Update device properties
            if device_id in self.device_registry:
                properties = self.device_registry[device_id].setdefault('properties', {})
                
                if state is not None:
                    properties['state'] = state
                
                if brightness is not None:
                    properties['brightness'] = brightness
                
                if color is not None:
                    properties['color'] = color
                
                self._save_device_registry()
    
    def _handle_lock_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a lock device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract lock state
        state = data.get('state')
        
        if state is not None:
            logger.debug(f"Lock {device_id} state: {state}")
            
            # Update device properties
            if device_id in self.device_registry:
                self.device_registry[device_id].setdefault('properties', {})['state'] = state
                self._save_device_registry()
    
    def _handle_speaker_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a speaker device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        # Extract speaker state
        state = data.get('state')
        volume = data.get('volume')
        
        if state is not None or volume is not None:
            logger.debug(f"Speaker {device_id} data: state={state}, volume={volume}")
            
            # Update device properties
            if device_id in self.device_registry:
                properties = self.device_registry[device_id].setdefault('properties', {})
                
                if state is not None:
                    properties['state'] = state
                
                if volume is not None:
                    properties['volume'] = volume
                
                self._save_device_registry()
    
    def _handle_generic_data(self, device_id: str, data: Dict[str, Any]) -> None:
        """
        Handle data from a generic device
        
        Args:
            device_id: Device ID
            data: Device data
        """
        logger.debug(f"Generic device {device_id} data: {data}")
        
        # Update device properties if state is provided
        if 'state' in data and device_id in self.device_registry:
            self.device_registry[device_id].setdefault('properties', {})['state'] = data['state']
            self._save_device_registry()

# Function to initialize and register the IoT features with the Flask app
def init_iot_features(app: Flask, config: Dict[str, Any], analytics_manager=None, adaptive_learning_manager=None) -> IoTManager:
    """Initialize and register IoT features with the Flask app"""
    iot_manager = IoTManager(app, config, analytics_manager, adaptive_learning_manager)
    return iot_manager
