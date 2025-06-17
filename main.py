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

# Custom CSS for animations and styling
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .match-container {
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        background-color: #f8f9fa;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 4px solid #4e79a7;
    }
    
    .match-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .progress-bar {
        height: 8px;
        background: #ecf0f1;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    .progress-value {
        height: 100%;
        background: linear-gradient(90deg, #4e79a7, #76b7b2);
        border-radius: 4px;
    }
    
    .search-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .stTextInput>div>div>input {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    .title-text {
        color: #2c3e50;
        font-weight: 700;
    }
    
    .subtitle-text {
        color: #7f8c8d;
        font-size: 1rem;
    }
    
    .stSpinner>div {
        border-color: #4e79a7 transparent transparent transparent !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .debug-info {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #4e79a7;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("<h1 class='title-text fade-in'>🔍 Find a GICC Professional</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p class='subtitle-text fade-in'>
            Describe the professional you need (e.g., "Accountant", "Driver", "Chef").<br>
            We'll find the best matches for your requirements.
        </p>
    """, unsafe_allow_html=True)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774270.png", width=100, caption="GICC Finder")



# Search box with enhanced styling
with stylable_container(
    key="search_box",
    css_styles="""
        {
            background-color: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
    """,
):
    user_query = st.text_input(
        "What kind of professional are you looking for?",
        placeholder="e.g., marketing consultant, electrician, web designer...",
        key="search_input"
    )

# Load model and data
model = load_model()
df = load_data()

# Add threshold slider in sidebar
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

        time.sleep(0.5)  # Reduced delay for better UX

    if top_results:
        st.success(f"✨ Found {len(top_results)} qualified professional(s):")
        
        # Display results
        for i, result in enumerate(top_results):
            score = result['score']
            row = result['row']
            
            # Create a container for each match with custom styling
            with stylable_container(
                key=f"match_{i}_{result['index']}",  # Make key unique
                css_styles="""
                    {
                        border-radius: 10px;
                        padding: 1.2rem;
                        margin: 0.8rem 0;
                        background-color: #f8f9fa;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        transition: all 0.3s ease;
                        border-left: 4px solid #4e79a7;
                    }
                """,
            ):
                # Display the professional details
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"🏆 #{i+1} - {row['NAME']}")
                    st.write(f"📞 **Phone:** 0{row['NUMBER']}")
                    st.write(f"📧 **Email**  {row['EMAIL']}")
                    st.write(f"💼 **Profession:** {row['PROFESSION']}")
                
                with col2:
                    # Visual match score indicator
                    st.metric("Match Score", f"{score:.0%}")
                
                # Progress bar
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-value" style="width: {score*100}%"></div>
                </div>
                """, unsafe_allow_html=True)
                
                
    else:
        st.error("🤔 No matching professionals found. Try:")
        st.write("• Using different keywords")
        st.write("• Lowering the match threshold")
        st.write("• Using broader terms")
        
        '''if debug_mode:
            st.write("Available professions in database:")
            unique_professions = df['PROFESSION'].dropna().unique()
            st.write(", ".join(unique_professions[:20]))  # Show first 20
            if len(unique_professions) > 20:
                st.write(f"... and {len(unique_professions) - 20} more")'''

# Add some empty space at the bottom
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# Optional: Add a floating action button
html("""
<script>
// Add a floating button
const floatingBtn = document.createElement('div');
floatingBtn.innerHTML = '<button style="position: fixed; bottom: 20px; right: 20px; background: #4e79a7; color: white; border: none; border-radius: 50%; width: 60px; height: 60px; font-size: 24px; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display: flex; align-items: center; justify-content: center;">💬</button>';
document.body.appendChild(floatingBtn);

// Add pulse animation to the button
setInterval(() => {
    const btn = floatingBtn.querySelector('button');
    if (btn) btn.classList.toggle('pulse');
}, 4000);
</script>
""")