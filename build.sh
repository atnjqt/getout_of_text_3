
#!/bin/bash

# Always run from project root
cd "$(dirname "$0")"

# Print ASCII art
echo -e '''
    __________________________________________________________________________________

   ______________________  __  ________    ____  ______  _____________  _____________
  / ____/ ____/_  __/ __ \/ / / /_  __/   / __ \/ ____/ /_  __/ ____/ |/ /_  __/__  /
 / / __/ __/   / / / / / / / / / / /_____/ / / / /_______/ / / __/  |   / / /   /_ < 
/ /_/ / /___  / / / /_/ / /_/ / / /_____/ /_/ / __/_____/ / / /___ /   | / /  ___/ / 
\____/_____/ /_/  \____/\____/ /_/      \____/_/       /_/ /_____//_/|_|/_/  /____/  
                                                                                     
__________________________________________________________________________________
'''

echo -e "\n"
current_version=$(awk -F'"' '/^version = / {print $2}' pyproject.toml)
echo "  getout_of_text_3 current version is: $current_version ⭐️"
echo -e "  Semver is major.minor.patch -- meaning you should bump the version according to the changes made if there are breaking changes or not!\n"

read -p "  👉 Enter the new version (semver): " version
read -p "  👉 Confirm version change from $current_version to $version? (y/n): " confirm
if [ "$confirm" != "y" ]; then
  echo "  👉 Version change aborted. ❌"
  exit 1
fi

# Update pyproject.toml
sed -i.bak "s/version = \"$current_version\"/version = \"$version\"/" pyproject.toml

# Update _version.py first line
sed -i.bak "1s/.*/__version__ = \"$version\"/" getout_of_text_3/_version.py

# Clean old builds
rm -rf dist/
rm -rf build/
rm -rf getout_of_text_3.egg-info/

echo "    Version updated successfully... ✅"
echo "    Building the package 🏗️"

# Build the package and check for success
if python3.11 -m build; then
    echo -e "\n"
    echo "    Package built successfully... ✅"
    echo -e "\n"
    
    # Verify dist files exist
    if [ ! -d "dist" ] || [ -z "$(ls -A dist/)" ]; then
        echo "    ❌ Error: dist/ directory is empty or doesn't exist!"
        exit 1
    fi
    
    sleep 0.5
else
    echo -e "\n"
    echo "    ❌ Package build failed!"
    exit 1
fi

# Prompt to publish
read -p "  👉 Do you want to publish the package? (y/n): " publish
if [ "$publish" == "y" ]; then
  echo -e "\n"
  echo "    Publishing the package... 🥳"
  sleep 1
  if [ -f "$HOME/.pypirc" ]; then
    echo "    Using credentials from $HOME/.pypirc"
    echo -e "\n"
    
    python3.11 -m twine upload dist/*
  else
    echo "    No .pypirc file found. Using manual authentication."
    python3.11 -m twine upload dist/*
  fi
else
  echo "  👉 Package not published. ❌"
fi