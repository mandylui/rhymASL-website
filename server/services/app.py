import streamlit as st
import os
from utils import *
import time

# Set page layout
st.set_page_config(layout="wide")

if "model_loaded" not in st.session_state:
    asl_list, entry_list,lemma_list = get_gloss_from_sentence("where is the restaurant")
    print(f"{asl_list}, {entry_list}. Model loaded successfully.")
    st.session_state.model_loaded = True


def render_header():
    # Use cached_story as the source of truth for display, not user_input
    story = st.session_state.get("cached_story", "")
    
    # Only compute if not already cached
    current_input = st.session_state.get("user_input", "")
    if (current_input and ("asl_list" not in st.session_state or st.session_state.get("cached_story") != current_input)):
        if 'related_signs_lemma_entryID' not in st.session_state:
            st.session_state.related_signs_lemma_entryID = dict()
        st.session_state.asl_list, st.session_state.entry_list,st.session_state.lemma_list = get_gloss_from_sentence(current_input,related=st.session_state.related_signs_lemma_entryID)
        st.session_state.video_path_list, st.session_state.clip_duration_list, st.session_state.lemma_id_list = get_sign_videos(st.session_state.entry_list)
        st.session_state.related_signs_lemma_entryID = dict()  # Reset to empty dict, not list
        
        st.session_state.cached_story = current_input  # mark which story is cached
        story = current_input  # Update story for display
        print(f"Done processing story: {st.session_state.cached_story}, ASL list: {st.session_state.asl_list}, Entry list: {st.session_state.entry_list}, Video paths: {st.session_state.video_path_list}")

    # Only proceed with display if we have processed data
    if "asl_list" not in st.session_state or not story:
        st.warning("Please enter a story first.")
        return
    
    # Display story and ASL gloss together after processing is complete
    st.markdown(f"### Story:\n\n**{story}**")
    
    asl_list = st.session_state.asl_list
    st.markdown("### ASL Gloss:")
    if asl_list:
        click_cols = st.columns(len(asl_list))
        for i, gloss in enumerate(asl_list):
            with click_cols[i]:
                if st.button(gloss, key=f"gloss_btn_{i}"):
                    st.session_state.clicked_word = gloss
                    st.session_state.page = 3
                    st.rerun()

    if st.button("▶️ Watch Whole Story"):
        st.session_state.autoplay = True
        st.session_state.force_autoplay = True  # Force autoplay flag
        st.rerun()  # Stay on page 2, just trigger autoplay



# --- App configuration ---

st.title("Phonological Sign Recommender")

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = 1
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "pending_story" not in st.session_state:
    st.session_state.pending_story = None

# Handle pre-selection before rendering the input widget
if st.session_state.pending_story:
    st.session_state.user_input = st.session_state.pending_story
    st.session_state.pending_story = None
    st.session_state.page = 2
    st.rerun()

# --- Page 1: Story Selection ---
if st.session_state.page == 1:
    st.header("Story Selection")

    st.text_input("Enter your own story line:", key="user_input")

    st.markdown("#### Or choose a pre-set:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("red worm, orange bison, yellow whale"):
            st.session_state.pending_story = "red worm, orange bison, yellow whale"
            st.session_state.page = 2
            st.session_state.autoplay = True
            st.rerun()
    with col2:
        if st.button("where is the restaurant"):
            st.session_state.pending_story = "where is the restaurant"
            st.session_state.page = 2
            st.session_state.autoplay = True
            st.rerun()

    if st.button("Submit Custom Story"):
        if st.session_state.user_input.strip():
            st.session_state.page = 2
            st.session_state.autoplay = True  # Fix #2: Set autoplay when submitting custom story
            st.rerun()


elif st.session_state.page == 2:
    # Always render header first to show story
    render_header()
    st.title("▶️ Full Story Playback")

    # Check if we have the required data
    if ("video_path_list" not in st.session_state or 
        "clip_duration_list" not in st.session_state or 
        "lemma_id_list" not in st.session_state):
        st.error("Story data not found. Please go back and select a story.")
        if st.button("← Back to Story Selection"):
            st.session_state.page = 1
            st.rerun()
    else:
        video_path_list = st.session_state.video_path_list
        clip_duration_list = st.session_state.clip_duration_list
        lemma_id_list = st.session_state.lemma_id_list
        print(f"Video paths: {video_path_list}, Clip durations: {clip_duration_list}, Lemma IDs: {lemma_id_list}")
        
        # Fix 1: Use force_autoplay to ensure proper autoplay behavior
        if st.session_state.get("autoplay", False) or st.session_state.get("force_autoplay", False):
            # Create a smaller video display area using columns
            col1, col2, col3 = st.columns([1, 2, 1])  # Center column is 2/4 of width
            
            with col2:  # Use the center column for video display
                st.write("🎬 Playing full story...")
                video_slot = st.empty()
                text_slot = st.empty()
                
                for i, video_path in enumerate(video_path_list):
                    if os.path.exists(video_path):
                        # Clear previous content completely
                        video_slot.empty()
                        text_slot.empty()
                        
                        # Small delay to ensure clearing
                        time.sleep(0.1)
                        
                        # Display current video and text
                        with video_slot.container():
                            st.video(video_path, format="video/mp4", start_time=0, autoplay=True)
                        with text_slot.container():
                            st.markdown(f"**Playing: {lemma_id_list[i]}**")
                        
                        # Wait for video duration
                        time.sleep(clip_duration_list[i]/1000 + 0.8)  # Longer buffer for last video
                
                # After completion, show a static end screen with image placeholder
                video_slot.empty()
                text_slot.empty()
                
                # Show completion screen with image placeholder
                with video_slot.container():
                    # Placeholder for random image - you can replace this with st.image() when you have an image
                    st.markdown("""
                    <div style="
                        border: 2px dashed #ccc; 
                        border-radius: 10px; 
                        padding: 40px; 
                        text-align: center; 
                        background-color: #f9f9f9;
                        margin: 20px 0;
                    ">
                        <h3>📸 Image Placeholder</h3>
                        <p>Replace this with: st.image('your_image_path.jpg', use_container_width=True)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("🎬 Story playback completed! Click 'Watch Whole Story' to play again.")
                with text_slot.container():
                    st.markdown("**✅ All signs played successfully**")
            
            # Clear both autoplay flags after playing
            st.session_state.autoplay = False
            st.session_state.force_autoplay = False

elif st.session_state.page == 3:
    if "clicked_word" not in st.session_state:
        st.error("No word selected. Please go back and click on a word.")
        if st.button("← Back to Story"):
            st.session_state.page = 2
            st.rerun()
    else:
        clicked = st.session_state.clicked_word
        st.title(f"🔍 Explore Sign: {clicked}")

        print(f"Clicked word: {clicked}, Entry ID: {st.session_state.entry_list}, Lemma ID: {st.session_state.lemma_list}, Video Path: {st.session_state.video_path_list}")
        # Get video for the clicked word
        # try:
        # extract the path from st.session_state
        clicked_word_index = st.session_state.asl_list.index(clicked)
        print("check1")
        clicked_entryID = st.session_state.entry_list[clicked_word_index]
        print("check2")
        clicked_video_path = st.session_state.video_path_list[clicked_word_index]
        print("check3")
        clicked_lemmaID = st.session_state.lemma_id_list[clicked_word_index]
        print("check4")
        clicked_sign_info = get_sign_info(clicked_entryID)
        st.write(f"handshape: {clicked_sign_info['handshape']}, movement: {clicked_sign_info['movement']}, location: {clicked_sign_info['major_location']}")
        print(f"Clicked Entry ID: {clicked_entryID}, Video Path: {clicked_video_path}, Lemma ID: {clicked_lemmaID}")
        
        # Create smaller video display area for individual sign
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(clicked_video_path, format="video/mp4", start_time=0, autoplay=True, loop=True)
            # caption
            st.markdown(f"**{clicked_lemmaID}**")
        # except Exception as e:
        #     st.error(f"Could not load video for '{clicked}': {e}")

        aspect = st.selectbox("Choose phonological aspect", ["handshape", "movement", "location", "mix"])
        k = st.slider("Number of related signs", 1, 5, 3)
        
        # Fix 3: Generate story only when selections are made and not already cached
        story_key = f"{clicked_entryID}_{aspect}_{k}"
        
        try:
            related_signs_entryID, related_signs_lemmaID = find_sign(clicked_entryID, aspect, k)
            print(f"Related signs found: {related_signs_entryID}, Lemma IDs: {related_signs_lemmaID}")
            related_video_path_list, related_clip_duration_list, _ = get_sign_videos(related_signs_entryID)
            print(f"Related video paths: {related_video_path_list}, Clip durations: {related_clip_duration_list}")

            st.markdown("### Related Signs:")
            # Create a grid layout for related signs videos
            cols_per_row = 2  # Adjust this to change how many videos per row
            for i in range(0, len(related_signs_entryID), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(related_signs_entryID) and idx < len(related_video_path_list):
                        with cols[j]:
                            st.video(related_video_path_list[idx], format="video/mp4", start_time=0)
                            st.markdown(f"**{related_signs_lemmaID[idx] if idx < len(related_signs_lemmaID) else 'N/A'}**")

            # Generate story only if not already cached for this configuration
            if f"generated_story_{story_key}" not in st.session_state:
                # Fixed the append issue
                input_words = related_signs_lemmaID.copy()  # Create a copy
                input_words.append(clicked_lemmaID)  # Now append returns the modified list
                
                new_story = generate_story(input_words)
                st.session_state[f"generated_story_{story_key}"] = new_story
            else:
                new_story = st.session_state[f"generated_story_{story_key}"]
                
            st.markdown(f"### ✏️ Generated Story:\n\n**{new_story}**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧠 Replace with New Story"):
                    related_signs_lemma_entryID = dict()
                    for i in range(len(related_signs_lemmaID)):
                        related_signs_lemma_entryID[related_signs_lemmaID[i]] = related_signs_entryID[i]
                    st.session_state.related_signs_lemma_entryID = related_signs_lemma_entryID
                    st.session_state.user_input = new_story
                    st.session_state.page = 2
                    st.session_state.autoplay = True
                    st.session_state.force_autoplay = True  # Ensure autoplay works
                    # Clear cached data to force reprocessing
                    if "cached_story" in st.session_state:
                        del st.session_state.cached_story
                    st.rerun()
            with col2:
                if st.button("↩️ Return to Current Story"):
                    st.session_state.page = 2
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error processing related signs: {e}")

# Add a back button on pages 2 and 3
if st.session_state.page in [2, 3]:
    if st.sidebar.button("← Back to Story Selection"):
        # Clear all cached data when going back
        keys_to_clear = ['asl_list', 'entry_list', 'lemma_list', 'video_path_list', 
                        'clip_duration_list', 'lemma_id_list', 'cached_story', 
                        'clicked_word', 'autoplay', 'force_autoplay']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear any generated story cache
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith('generated_story_')]
        for key in keys_to_remove:
            del st.session_state[key]
            
        st.session_state.page = 1
        st.rerun()