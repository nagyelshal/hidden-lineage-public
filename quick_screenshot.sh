#!/bin/bash
echo "📸 Taking screenshot of dashboard..."

# For systems with gnome-screenshot
if command -v gnome-screenshot &> /dev/null; then
    gnome-screenshot -w -f exports/social_media/dashboard_window.png
    echo "✅ Screenshot saved using gnome-screenshot"
# For systems with scrot
elif command -v scrot &> /dev/null; then
    scrot -s exports/social_media/dashboard_selection.png
    echo "✅ Screenshot saved using scrot"
else
    echo "📱 Please take manual screenshots:"
    echo "   1. Open: http://localhost:8502"
    echo "   2. Use browser screenshot tools"
    echo "   3. Save to: exports/social_media/"
fi
