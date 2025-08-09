---
description: Create a pull request following project best practices
allowed-tools: [Bash, Read, Glob, Grep]
argument-hint: "[optional pr title]"
---

# Pull Request Command

Create well-structured pull requests.

## Instructions

1. **Review branch changes**: 
   - Check git status and current branch
   - Review all commits that will be included in the PR
   - Ensure branch is up to date with staging
3. **Create PR**:
   - Use descriptive title following project patterns
   - Include clear summary of changes in description
   - Explain what problem the PR solves and provide context
4. **Push and create**: Push branch to remote and create PR using gh CLI

If user provided a PR title as $ARGUMENTS, use it (ensure it follows project naming patterns).

Example PR titles:
- `feat: Add KnowledgeBaseService with text extraction capabilities`

Always include context about what the PR accomplishes and why the changes were needed.