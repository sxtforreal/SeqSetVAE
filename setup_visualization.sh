#!/bin/bash
# Setup script for recon eval visualization tools

echo "🚀 Setting up Recon Eval Visualization Tools"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "evaluate_reconstruction.py" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Try to create virtual environment
echo "📦 Setting up Python environment..."
if command -v python3 &> /dev/null; then
    if python3 -m venv viz_env 2>/dev/null; then
        echo "✅ Virtual environment created: viz_env"
        source viz_env/bin/activate
        
        echo "📥 Installing visualization dependencies..."
        pip install numpy matplotlib seaborn pandas scikit-learn
        
        if [ $? -eq 0 ]; then
            echo "✅ Dependencies installed successfully!"
            echo ""
            echo "🎨 Testing visualization tools..."
            python demo_visualizer.py
            echo ""
            echo "✅ Setup complete! To use:"
            echo "   source viz_env/bin/activate"
            echo "   python recon_eval_visualizer.py --help"
        else
            echo "❌ Failed to install dependencies"
            echo "💡 You can still use simple_json_viewer.py (no dependencies required)"
        fi
    else
        echo "⚠️  Could not create virtual environment"
        echo "💡 Using system packages or simple viewer instead"
    fi
else
    echo "❌ Python3 not found"
    exit 1
fi

echo ""
echo "📋 Available tools:"
echo "  • simple_json_viewer.py  (no dependencies, text-based analysis)"
echo "  • recon_eval_visualizer.py (rich visualizations, requires dependencies)"
echo ""
echo "📖 See VISUALIZATION_GUIDE.md for detailed usage instructions"