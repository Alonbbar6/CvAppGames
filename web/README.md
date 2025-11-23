# Face Control Games - Web Version

Play classic games using your face! Built with TensorFlow.js Face Landmarks Detection.

## Games Included

1. **Flappy Bird** - Raise your eyebrows to make the bird flap
2. **Tetris** - Move your head left/right, raise eyebrows to rotate
3. **Expression Match** - Match facial expressions to score points

## Deploy to Netlify

### Option 1: Drag and Drop

1. Go to [Netlify](https://app.netlify.com)
2. Sign up or log in
3. Drag and drop the entire `web` folder onto the Netlify dashboard
4. Your site will be live in seconds!

### Option 2: Git Deployment

1. Push this `web` folder to a GitHub repository
2. Connect your GitHub account to Netlify
3. Select the repository and set:
   - Build command: (leave empty)
   - Publish directory: `web` (or `.` if web folder is the root)
4. Click Deploy

### Option 3: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy from the web folder
cd web
netlify deploy --prod
```

## Local Testing

To test locally before deploying:

```bash
# Using Python
cd web
python3 -m http.server 8000

# Or using Node.js
npx serve .
```

Then open http://localhost:8000 in your browser.

**Note:** Camera access requires HTTPS in production. Localhost is an exception for testing.

## Browser Requirements

- Modern browser (Chrome, Firefox, Safari, Edge)
- Camera access permission
- WebGL support (for TensorFlow.js)

## Privacy

All face detection happens locally in your browser. No video data is sent to any server.
