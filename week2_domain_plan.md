# Domain Blueprint: Linux Command Assistant
**Author:** Avirup Ghosh

## 1. Target Audience & Use Case
* **Audience**: Students, beginners, and Linux users who want to learn Linux commands and system administration.
* **Problem Statement**: Linux documentation is extensive and can be difficult to navigate, especially for beginners. This RAG-based Linux Documentation Assistant retrieves relevant information from trusted Linux resources to provide accurate, context-aware answers quickly.

## 2. Expected User Queries (Five Samples)
1. "How do I create, copy, move, and delete files or directories in Linux?"
2. "How do I change file permissions and ownership using chmod and chown?"
3. "How can I monitor running processes and stop a process in Linux?"
4. "How do I configure or troubleshoot a network connection using Linux commands?"
5. "How do I write and execute a Bash shell script?"

## 3. Data Sources (Knowledge Base)
* Trusted Linux resources, official command manuals, and system administration guides.

## 4. System Constraints & Boundary Rules (Out of Scope)
* **Unrelated Topics**: Questions unrelated to Linux.
* **Document Boundary**: Questions that cannot be answered using the uploaded document.
* **Malicious Content**: Requests involving hacking, malware, or illegal activities.
* **Non-Technical Advice**: Personal, medical, legal, or financial advice.
* **Speculation**: Questions requiring personal opinions, assumptions, or unsupported speculation.

## 5. Success Criteria
* Correctly explain Linux commands.
* Explain the Linux file system, including directories.
* Help users write and understand Bash shell scripts using variables, loops, conditions, and functions.
* Retrieve the correct information from the uploaded Linux documents and provide accurate answers within a few seconds.
* Avoids generating unsupported answers when information is unavailable.