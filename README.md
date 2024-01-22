---
title: UOC Library Chat App V1
---


#create virtual environment
python -m venv ./env

#activate environment
env/Scripts/Activate.ps1

#pip install
pip install --upgrade pip  
pip install requirements.txt

#How to run streamlit app
python -m server  

#How to run server
python -m streamlit run app.py
  

repository branches naming conventions
Some suggestions for naming feature branches:

* users/username/description
* users/username/workitem
* bugfix/description
* feature/feature-name
* feature/feature-area/feature-name
* hotfix/description
* release/v1.0.1

Branch Prefixes
1. Feature Branches: These branches are used for developing new features. Use the prefix feature/. For instance, feature/login-system.
2. Bugfix Branches: These branches are used to fix bugs in the code. Use the prefix bugfix/. For example, bugfix/header-styling.
3. Hotfix Branches: These branches are made directly from the production branch to fix critical bugs in the production environment. Use the prefix hotfix/. For instance, hotfix/critical-security-issue.
4. Release Branches: These branches are used to prepare for a new production release. They allow for last-minute dotting of i’s and crossing t’s. Use the prefix release/. For example, release/v1.0.1.
5. Documentation Branches: These branches are used to write, update, or fix documentation. Use the prefix docs/. For instance, docs/api-endpoints.

Basic Rules
1. Lowercase and Hyphen-separated: Stick to lowercase for branch names and use hyphens to separate words. For instance, feature/new-login or bugfix/header-styling.
2. Alphanumeric Characters: Use only alphanumeric characters (a-z, 0–9) and hyphens. Avoid punctuation, spaces, underscores, or any non-alphanumeric character.
3. No Continuous Hyphens: Do not use continuous hyphens. feature--new-login can be confusing and hard to read.
4. No Trailing Hyphens: Do not end your branch name with a hyphen. For example, feature-new-login- is not a good practice.
5. Descriptive: The name should be descriptive and concise, ideally reflecting the work done on the branch.