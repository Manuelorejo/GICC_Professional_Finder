# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:33:33 2025

@author: Oreoluwa
"""

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import time
from streamlit.components.v1 import html
from streamlit_extras.stylable_container import stylable_container

# Load the model only once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load data only once
@st.cache_data
def load_data():    
    return pd.read_csv('demo.csv')

# Apply custom page config with wider layout
st.set_page_config(
    page_title="GICC Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme Detection and Custom CSS
st.markdown("""
<style>
    /* CSS Variables for theming */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-card: #ffffff;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --border-color: #e1e5e9;
        --shadow-light: rgba(0,0,0,0.1);
        --shadow-medium: rgba(0,0,0,0.15);
        --accent-primary: #4e79a7;
        --accent-secondary: #76b7b2;
        --success-bg: #d4edda;
        --error-bg: #f8d7da;
        --progress-bg: #ecf0f1;
    }

    /* Dark theme variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --bg-card: #1e1e2e;
            --text-primary: #fafafa;
            --text-secondary: #c9c9c9;
            --border-color: #3d3d3d;
            --shadow-light: rgba(255,255,255,0.1);
            --shadow-medium: rgba(255,255,255,0.15);
            --accent-primary: #6ba3d6;
            --accent-secondary: #76b7b2;
            --success-bg: #1e3a28;
            --error-bg: #3a1e20;
            --progress-bg: #404040;
        }
    }

    /* Manual dark theme class (when Streamlit dark mode is enabled) */
    .stApp[data-theme="dark"] {
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --bg-card: #1e1e2e;
        --text-primary: #fafafa;
        --text-secondary: #c9c9c9;
        --border-color: #3d3d3d;
        --shadow-light: rgba(255,255,255,0.1);
        --shadow-medium: rgba(255,255,255,0.15);
        --accent-primary: #6ba3d6;
        --accent-secondary: #76b7b2;
        --success-bg: #1e3a28;
        --error-bg: #3a1e20;
        --progress-bg: #404040;
    }

    /* Base animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }

    /* Responsive card styling */
    .match-container {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: var(--bg-card);
        box-shadow: 0 4px 12px var(--shadow-light);
        transition: all 0.3s ease;
        border-left: 4px solid var(--accent-primary);
        border: 1px solid var(--border-color);
    }
    
    .match-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px var(--shadow-medium);
    }

    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .match-container {
            padding: 1rem;
            margin: 0.8rem 0;
            border-radius: 8px;
        }
        
        .search-box {
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        .title-text {
            font-size: 1.8rem !important;
        }
        
        .subtitle-text {
            font-size: 0.9rem !important;
        }
    }

    @media (max-width: 480px) {
        .match-container {
            padding: 0.8rem;
            margin: 0.5rem 0;
        }
        
        .title-text {
            font-size: 1.5rem !important;
        }
        
        .floating-btn {
            width: 50px !important;
            height: 50px !important;
            font-size: 20px !important;
            bottom: 15px !important;
            right: 15px !important;
        }
    }
    
    /* Progress bar styling */
    .progress-bar {
        height: 8px;
        background: var(--progress-bg);
        border-radius: 4px;
        margin: 12px 0;
        overflow: hidden;
    }
    
    .progress-value {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Search box styling */
    .search-box {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px var(--shadow-light);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 8px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(78, 121, 167, 0.1) !important;
    }
    
    /* Typography */
    .title-text {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: var(--text-secondary);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Debug info styling */
    .debug-info {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid var(--accent-primary);
        border: 1px solid var(--border-color);
    }
    
    /* Spinner customization */
    .stSpinner>div {
        border-color: var(--accent-primary) transparent transparent transparent !important;
    }
    
    /* Success and error messages */
    .element-container .stSuccess {
        background-color: var(--success-bg);
        border: 1px solid var(--accent-secondary);
        border-radius: 8px;
    }
    
    .element-container .stError {
        background-color: var(--error-bg);
        border: 1px solid #dc3545;
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
    }
    
    /* Metric styling */
    .metric-container {
        background-color: var(--bg-secondary);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    /* Professional card content */
    .professional-header {
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .professional-details {
        color: var(--text-secondary);
        margin: 0.3rem 0;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .professional-details strong {
        color: var(--text-primary);
        margin-right: 0.5rem;
    }
    
    /* Responsive grid adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Stack columns on mobile */
        .professional-card-mobile {
            flex-direction: column;
        }
        
        .professional-details {
            font-size: 0.9rem;
        }
    }
    
    /* Image responsiveness */
    .header-image {
        max-width: 100%;
        height: auto;
    }
    
    @media (max-width: 768px) {
        .header-image {
            width: 80px;
        }
    }
    
    /* Floating button responsive */
    .floating-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 12px var(--shadow-medium);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .floating-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px var(--shadow-medium);
    }
    
    /* Dark mode toggle button */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 12px;
        z-index: 1001;
        transition: all 0.3s ease;
    }
    
    .theme-toggle:hover {
        background: var(--accent-secondary);
    }
    
    /* Loading state improvements */
    .search-loading {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
    }
    
    /* Enhanced card interactions */
    .match-container:active {
        transform: translateY(-1px);
    }
    
    /* Better spacing for mobile */
    @media (max-width: 768px) {
        .element-container {
            margin-bottom: 1rem;
        }
        
        .stColumns > div {
            padding: 0 0.5rem;
        }
    }
    
    .block-container {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

# Theme detection script
html("""
<script>
function detectTheme() {
    const streamlitDoc = window.parent.document;
    const streamlitRoot = streamlitDoc.querySelector('.stApp');
    
    // Check if Streamlit dark mode is active
    const isDarkMode = streamlitRoot.getAttribute('data-theme') === 'dark' || 
                      streamlitDoc.body.classList.contains('dark') ||
                      window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (isDarkMode && !streamlitRoot.classList.contains('dark-theme')) {
        streamlitRoot.classList.add('dark-theme');
    }
}

// Run theme detection
detectTheme();

// Listen for theme changes
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectTheme);
}
</script>
""")

# Responsive header with better mobile layout
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown("<h1 class='title-text fade-in'>🔍 Find a GICC Professional</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p class='subtitle-text fade-in'>
            Describe the professional you need (e.g., "Accountant", "Driver", "Chef").<br>
            We'll find the best matches for your requirements.
        </p>
    """, unsafe_allow_html=True)

with header_col2:
    # Responsive image
    st.markdown("""
        <img src="https://cdn-icons-png.flaticon.com/512/3774/3774270.png" 
             class="header-image" 
             style="width: 100px; height: auto;" 
             alt="GICC Finder">
        <p style="text-align: center; font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">GICC Finder</p>
    """, unsafe_allow_html=True)

# Enhanced search box with better mobile support
with stylable_container(
    key="search_box",
    css_styles="""
        {
            background-color: var(--bg-card);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px var(--shadow-light);
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
        }
    """,
):
    user_query = st.text_input(
        "What kind of professional are you looking for?",
        placeholder="e.g., marketing consultant, electrician, web designer...",
        key="search_input",
    )

# Load model and data
model = load_model()
df = load_data()

# Configuration
threshold = 0.5
max_results = 10

# Process query if input exists
if user_query:
    with st.spinner('🔍 Searching our network for the best professionals...'):
        try:
            # Encode database professions
            profession_texts = df['PROFESSION'].fillna('').astype(str).tolist()
            profession_embeddings = model.encode(profession_texts, convert_to_tensor=True)

            # Encode query
            query_embedding = model.encode(user_query, convert_to_tensor=True)

            # Compute cosine similarities
            cosine_scores = util.cos_sim(query_embedding, profession_embeddings)[0]

            # Create results with indices and scores
            all_results = []
            for idx, score in enumerate(cosine_scores):
                score_value = score.item()
                if score_value >= threshold:
                    all_results.append({
                        'score': score_value,
                        'index': idx,
                        'row': df.iloc[idx]
                    })
            
            # Sort by score (highest first)
            all_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit results
            top_results = all_results[:max_results]
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            top_results = []

        time.sleep(0.5)

    if top_results:
        st.success(f"✨ Found {len(top_results)} qualified professional(s):")
        
        # Display results with responsive design
        for i, result in enumerate(top_results):
            score = result['score']
            row = result['row']
            
            # Create responsive container for each match
            with stylable_container(
                key=f"match_{i}_{result['index']}",
                css_styles="""
                    {
                        border-radius: 12px;
                        padding: 1.5rem;
                        margin: 1rem 0;
                        background-color: var(--bg-card);
                        box-shadow: 0 4px 12px var(--shadow-light);
                        transition: all 0.3s ease;
                        border-left: 4px solid var(--accent-primary);
                        border: 1px solid var(--border-color);
                    }
                """,
            ):
                # Responsive layout: stack on mobile, side-by-side on desktop
                desktop_col1, desktop_col2 = st.columns([3, 1])
                
                with desktop_col1:
                    st.markdown(f"<h3 class='professional-header'>🏆 #{i+1} - {row['NAME']}</h3>", 
                              unsafe_allow_html=True)
                    
                    # Professional details with responsive styling
                    st.markdown(f"""
                        <div class='professional-details'>
                            <span><strong>📞 Phone:</strong> 0{row['NUMBER']}</span>
                        </div>
                        <div class='professional-details'>
                            <span><strong>📧 Email:</strong> {row['EMAIL']}</span>
                        </div>
                        <div class='professional-details'>
                            <span><strong>💼 Profession:</strong> {row['PROFESSION']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                with desktop_col2:
                    # Responsive metric display
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div style='font-size: 1.2rem; font-weight: bold; color: var(--accent-primary);'>
                                {score:.0%}
                            </div>
                            <div style='font-size: 0.8rem; color: var(--text-secondary);'>
                                Match Score
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Progress bar with enhanced styling
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-value" style="width: {score*100}%"></div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.error("🤔 No matching professionals found. Try:")
        suggestions_col1, suggestions_col2 = st.columns(2)
        
        with suggestions_col1:
            st.write("• Using different keywords")
            st.write("• Using broader terms")
        
        with suggestions_col2:
            st.write("• Try single word searches")
            st.write("• Check spelling")

# Add responsive spacing
st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

# Enhanced floating action button with responsive design
html("""
<script>
// Create responsive floating button
function createFloatingButton() {
    // Remove existing button if any
    const existingBtn = document.querySelector('.floating-btn');
    if (existingBtn) {
        existingBtn.remove();
    }
    
    const floatingBtn = document.createElement('button');
    floatingBtn.className = 'floating-btn';
    floatingBtn.innerHTML = '💬';
    floatingBtn.title = 'Need help? Contact support';
    
    // Add click handler
    floatingBtn.onclick = function() {
        alert('Contact support: manuelorejo@gmail.com');
    };
    
    document.body.appendChild(floatingBtn);
    
    // Add pulse animation periodically
    setInterval(() => {
        floatingBtn.classList.add('pulse');
        setTimeout(() => {
            floatingBtn.classList.remove('pulse');
        }, 2000);
    }, 8000);
}

// Create button when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createFloatingButton);
} else {
    createFloatingButton();
}

// Responsive adjustments based on screen size
function adjustForScreenSize() {
    const floatingBtn = document.querySelector('.floating-btn');
    if (floatingBtn) {
        if (window.innerWidth <= 480) {
            floatingBtn.style.width = '50px';
            floatingBtn.style.height = '50px';
            floatingBtn.style.fontSize = '20px';
            floatingBtn.style.bottom = '15px';
            floatingBtn.style.right = '15px';
        } else {
            floatingBtn.style.width = '60px';
            floatingBtn.style.height = '60px';
            floatingBtn.style.fontSize = '24px';
            floatingBtn.style.bottom = '20px';
            floatingBtn.style.right = '20px';
        }
    }
}

// Listen for resize events
window.addEventListener('resize', adjustForScreenSize);
adjustForScreenSize(); // Initial call

// Enhanced theme detection with better compatibility
function enhancedThemeDetection() {
    const streamlitDoc = window.parent.document;
    const streamlitRoot = streamlitDoc.querySelector('.stApp');
    
    if (streamlitRoot) {
        // Multiple ways to detect dark mode
        const isDark = streamlitRoot.getAttribute('data-theme') === 'dark' ||
                      streamlitDoc.documentElement.getAttribute('data-theme') === 'dark' ||
                      streamlitDoc.body.classList.contains('dark-mode') ||
                      window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (isDark) {
            streamlitRoot.setAttribute('data-theme', 'dark');
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            streamlitRoot.setAttribute('data-theme', 'light');
            document.documentElement.setAttribute('data-theme', 'light');
        }
    }
}

// Run enhanced theme detection
enhancedThemeDetection();

// Listen for system theme changes
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', enhancedThemeDetection);
}

// Periodic theme check (for Streamlit theme changes)
setInterval(enhancedThemeDetection, 1000);
</script>
""")
