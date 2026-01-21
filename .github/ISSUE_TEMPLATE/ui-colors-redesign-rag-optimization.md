---
name: UI Colors Redesign and RAG Optimization
about: Redesign UI theme to match presentation style and optimize RAG parameters for better content generation
title: '[FEATURE] UI Colors Redesign and RAG Optimization'
labels: enhancement, ui, rag
assignees: ''
---

## 📋 Description

This feature implements a comprehensive UI redesign to match the presentation color scheme and optimizes RAG (Retrieval-Augmented Generation) parameters for improved content quality and relevance.

## 🎨 UI Changes

### Color Scheme
- **Background**: Light blue gradient (#C8E0F4 → #A8C8E8) matching presentation bokeh effect
- **Sidebar**: Navy blue gradient (#2C3E5A → #16202E) with cyan accents
- **Primary Color**: Cyan (#00D4FF) for buttons and highlights
- **Accent Color**: Purple (#9F7AEA) for sliders and interactive elements
- **Text**: High contrast black (#0F172A) for titles, dark gray (#1E293B) for paragraphs

### Layout Improvements
- Compact header with centered title and project info
- Left-aligned content section heading for better space utilization
- Removed unnecessary dividers and blocks between sections
- Enhanced text contrast for better readability
- Info/warning blocks with consistent purple styling

### Input Fields
- Converted default values to placeholders that disappear when typing
- Applied to "Topic", "Audience", "Company Info", and "Research Topic" fields
- Improved placeholder visibility with darker gray (#475569) and full opacity

## 🔧 RAG Optimization

### Updated Parameters
| Parameter | Previous | New | Reason |
|-----------|----------|-----|--------|
| `chunk_size` | 600 | 1400 | Better context preservation, reduced information loss |
| `chunk_overlap` | 50 | 175 | Improved continuity between chunks |
| `max_results` | 2 | 7 | More comprehensive paper retrieval |
| `k` (retrieval) | 3 | 5 | Increased relevant document retrieval |

### Fixed Issues
- Removed `similarity_score_threshold` that was rejecting all documents due to ChromaDB returning negative distances
- Changed to simple `similarity` search without threshold
- Fixed metadata extraction bug in `get_context()` function

## 🔗 arXiv Source Links

### Implementation
- Added full metadata display for arXiv papers (title, authors, publication date, URL)
- Implemented expandable sections for each source
- Direct links to arXiv papers for reference verification
- Dual display format: URL text + clickable link button

### Benefits
- Improved academic credibility with proper source attribution
- Easy verification of information sources
- Enhanced user trust in generated content

## 🖼️ Image Generation Improvements

### Quality Enhancements
- Increased `guidance_scale` from 8.5 to 13 for better prompt adherence (no performance impact)
- Enhanced prompt engineering: 150-200 words vs previous 100 words
- Improved concept extraction: identifies 3-5 specific key concepts from article
- More detailed image descriptions including subject, environment, lighting, and composition

### Expected Results
- More relevant images matching article content
- Better visual representation of technical concepts
- Improved photorealistic quality

## 📁 Files Modified

- `app.py` - UI theme, layout, input fields, CSS styling
- `src/core/rag_engine.py` - RAG parameters, similarity search fix, metadata handling
- `config/prompts.py` - Image generation prompt template
- `src/models/image_generator.py` - Guidance scale parameter
- `.streamlit/config.toml` - Base theme colors

## ✅ Testing Checklist

- [ ] UI renders correctly in different screen sizes
- [ ] All color contrasts meet WCAG accessibility standards
- [ ] Placeholders disappear when user starts typing
- [ ] arXiv links open correctly and display full metadata
- [ ] RAG retrieves relevant papers with improved parameters
- [ ] Generated images are more relevant to article content
- [ ] No regression in existing functionality

## 📸 Screenshots

_To be added during PR review_

## 🚀 Deployment Notes

- No new dependencies required
- Docker rebuild necessary for changes to take effect
- No database migrations needed
- ChromaDB data will be preserved (no schema changes)

## 📝 Related Issues

Closes #XX (to be created)

## 👥 Reviewers

@[teammate-username] - Please review UI/UX changes and RAG parameter optimization
