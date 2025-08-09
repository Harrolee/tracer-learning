---
description: Create a git commit following project best practices
allowed-tools: [Bash, Read, Glob, Grep]
argument-hint: "[optional commit message]"
---

# Commit Command

Create atomic and well-formatted commits.

## Instructions

1. **Review current changes**: Check git status and diff to understand what will be committed
1. **Create atomic commit**: 
   - Use conventional commit format: `type: description`
   - Types: `feat:`, `fix:`, `chore:`
   - Keep commits focused on a single logical unit of work
   - Split large changes into multiple commits if needed
1. **Commit message**: Use clear, descriptive messages

If user provided a commit message as $ARGUMENTS, use it as the commit description (still following conventional commit format).

Always ensure the commit represents a complete, working change.