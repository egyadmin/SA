/**
 * Main JavaScript for Manus Clone
 * This script handles the UI interactions and communication with the backend
 */

// Import required modules
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const dotenv = require('dotenv');
const winston = require('winston');
const axios = require('axios');

// Load environment variables
dotenv.config();

// Initialize logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Initialize Express app
const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Configure middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Configure file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({ storage });

// In-memory data store (replace with database in production)
const conversations = [];
const messages = [];
const tasks = [];
const files = [];

// API Routes
// Conversations
app.get('/api/conversations', (req, res) => {
  res.json({ conversations });
});

app.post('/api/conversations', (req, res) => {
  const newConversation = {
    id: uuidv4(),
    summary: req.body.summary || 'New Conversation',
    last_updated: Date.now()
  };
  conversations.push(newConversation);
  res.status(201).json(newConversation);
});

app.get('/api/conversations/:id', (req, res) => {
  const conversation = conversations.find(c => c.id === req.params.id);
  if (!conversation) {
    return res.status(404).json({ error: 'Conversation not found' });
  }
  const conversationMessages = messages.filter(m => m.conversationId === req.params.id);
  res.json({ conversation, messages: conversationMessages });
});

app.delete('/api/conversations/:id', (req, res) => {
  const index = conversations.findIndex(c => c.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Conversation not found' });
  }
  conversations.splice(index, 1);
  // Also delete associated messages
  const messagesToDelete = messages.filter(m => m.conversationId === req.params.id);
  messagesToDelete.forEach(message => {
    const msgIndex = messages.findIndex(m => m.id === message.id);
    if (msgIndex !== -1) {
      messages.splice(msgIndex, 1);
    }
  });
  res.json({ success: true });
});

// Messages
app.post('/api/messages', (req, res) => {
  const newMessage = {
    id: uuidv4(),
    conversationId: req.body.conversationId,
    content: req.body.content,
    role: req.body.role || 'user',
    timestamp: Date.now(),
    attachments: req.body.attachments || []
  };
  messages.push(newMessage);
  
  // Update conversation last_updated
  const conversation = conversations.find(c => c.id === req.body.conversationId);
  if (conversation) {
    conversation.last_updated = Date.now();
    // Update summary if it's a new conversation
    if (conversation.summary === 'New Conversation') {
      conversation.summary = req.body.content.substring(0, 50) + (req.body.content.length > 50 ? '...' : '');
    }
  }
  
  // If it's a user message, generate AI response
  if (req.body.role === 'user') {
    // Simulate AI processing
    setTimeout(() => {
      const aiResponse = {
        id: uuidv4(),
        conversationId: req.body.conversationId,
        content: generateAIResponse(req.body.content),
        role: 'assistant',
        timestamp: Date.now(),
        attachments: []
      };
      messages.push(aiResponse);
      
      // Emit the new message via Socket.IO
      io.emit('new_message', aiResponse);
    }, 1000);
  }
  
  res.status(201).json(newMessage);
});

// File upload
app.post('/api/upload', upload.array('files'), (req, res) => {
  const uploadedFiles = req.files.map(file => {
    const fileRecord = {
      id: uuidv4(),
      filename: file.originalname,
      path: file.path,
      mimetype: file.mimetype,
      size: file.size,
      uploaded_at: Date.now()
    };
    files.push(fileRecord);
    return fileRecord;
  });
  
  res.status(201).json({ files: uploadedFiles });
});

// Tasks
app.get('/api/tasks', (req, res) => {
  res.json({ tasks });
});

app.post('/api/tasks', (req, res) => {
  const newTask = {
    id: uuidv4(),
    title: req.body.title,
    description: req.body.description || '',
    status: req.body.status || 'pending',
    created_at: Date.now(),
    updated_at: Date.now(),
    due_date: req.body.due_date || null
  };
  tasks.push(newTask);
  res.status(201).json(newTask);
});

app.put('/api/tasks/:id', (req, res) => {
  const taskIndex = tasks.findIndex(t => t.id === req.params.id);
  if (taskIndex === -1) {
    return res.status(404).json({ error: 'Task not found' });
  }
  
  tasks[taskIndex] = {
    ...tasks[taskIndex],
    ...req.body,
    updated_at: Date.now()
  };
  
  res.json(tasks[taskIndex]);
});

app.delete('/api/tasks/:id', (req, res) => {
  const taskIndex = tasks.findIndex(t => t.id === req.params.id);
  if (taskIndex === -1) {
    return res.status(404).json({ error: 'Task not found' });
  }
  
  tasks.splice(taskIndex, 1);
  res.json({ success: true });
});

// Socket.IO connection
io.on('connection', (socket) => {
  logger.info('New client connected');
  
  socket.on('disconnect', () => {
    logger.info('Client disconnected');
  });
  
  socket.on('chat_message', (message) => {
    // Process the message and generate a response
    const response = {
      id: uuidv4(),
      conversationId: message.conversationId,
      content: generateAIResponse(message.content),
      role: 'assistant',
      timestamp: Date.now()
    };
    
    // Emit the response
    io.emit('new_message', response);
  });
});

// Helper function to generate AI responses (placeholder)
function generateAIResponse(userMessage) {
  // In a real implementation, this would call the AI model
  const responses = [
    "أفهم ما تقوله. هل يمكنك توضيح المزيد؟",
    "هذا سؤال مثير للاهتمام. دعني أبحث عن إجابة لك.",
    "يمكنني مساعدتك في ذلك. هل تريد مني البدء الآن؟",
    "لقد فهمت طلبك وسأعمل عليه.",
    "شكراً لمشاركة هذه المعلومات. هل هناك أي شيء آخر تحتاجه؟"
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}

// Start the server
const PORT = process.env.PORT || 7860;
server.listen(PORT, '0.0.0.0', () => {
  logger.info(`Server running on port ${PORT}`);
  console.log(`Server running on port ${PORT}`);
});

// Export the app for testing
module.exports = app;
