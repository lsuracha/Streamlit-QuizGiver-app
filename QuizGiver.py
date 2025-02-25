import streamlit as st
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
import random
import re

# Initialize session state variables
def initialize_session_state():
    defaults = {
        'context': "",
        'question': "",
        'options': [],
        'correct_answer': "",
        'selected_option': None,
        'submitted': False,
        'question_count': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load the Hugging Face pipelines
@st.cache_resource
def load_models():
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    qg_tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    qg_pipeline = pipeline("text2text-generation", model=qg_model, tokenizer=qg_tokenizer)
    return qa_model, qg_pipeline

qa_model, qg_model = load_models()

# Function to generate question and smarter distractors
def generate_question_and_options(context):
    try:
        sentences = re.split(r'[.!?]\s+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 4:
            return None, None, None, "The context is too short. Please provide at least 4 sentences."

        target_sentence = random.choice(sentences)
        qg_input = f"answer: {target_sentence} context: {context}"
        question = qg_model(qg_input, max_length=128, num_beams=4)[0]['generated_text'].strip()
        result = qa_model(question=question, context=context)
        correct_answer = result['answer'].strip()

        # New distractor generation: Perturb the correct answer
        distractors = set()
        
        def perturb_answer(answer, context_sentences):
            words = answer.split()
            perturbed = []
            
            # 1. Change numbers if present
            if any(w.isdigit() for w in words):
                for i, w in enumerate(words):
                    if w.isdigit():
                        num = int(w)
                        perturbed.append(" ".join(words[:i] + [str(num + random.randint(1, 5))] + words[i+1:]))
                        perturbed.append(" ".join(words[:i] + [str(num - random.randint(1, 5))] + words[i+1:]))
            
            # 2. Swap a key noun with another from context
            other_nouns = []
            for s in context_sentences:
                for w in s.split():
                    if w.lower() not in answer.lower() and (w[0].isupper() or len(w) > 4):
                        other_nouns.append(w)
            if other_nouns:
                for i, w in enumerate(words):
                    if w[0].isupper() or len(w) > 4:
                        perturbed.append(" ".join(words[:i] + [random.choice(other_nouns)] + words[i+1:]))
            
            # 3. Fallback: Use QA on other sentences
            if len(perturbed) < 3:
                other_sentences = [s for s in sentences if s != target_sentence]
                for s in random.sample(other_sentences, min(2, 3 - len(perturbed))):
                    distractor = qa_model(question=question, context=s)['answer'].strip()
                    if distractor != answer and distractor not in perturbed:
                        perturbed.append(distractor)
            
            return perturbed[:3]

        distractors = perturb_answer(correct_answer, sentences)
        
        while len(distractors) < 3:
            fallback = random.choice(["None", "Unknown", "Other"])
            if fallback not in distractors and fallback != correct_answer:
                distractors.append(fallback)

        options = distractors + [correct_answer]
        random.shuffle(options)
        return question, options, correct_answer, None
    except Exception as e:
        return None, None, None, f"Error generating question: {str(e)}"

# Streamlit app layout
st.title("📝 Enhanced Multiple-Choice Question Generator")

initialize_session_state()

with st.form(key='input_form'):
    context = st.text_area("Enter the context (e.g., a paragraph or document):", 
                          value=st.session_state.context,
                          height=200)
    submit_context = st.form_submit_button(label="Generate Question")

col1, col2, col3 = st.columns(3)
with col1:
    generate_another = st.button("Generate Another Question")
with col2:
    reset = st.button("Reset")
with col3:
    pass

if reset:
    st.session_state.clear()
    initialize_session_state()
    st.rerun()

if submit_context or (generate_another and st.session_state.context):
    if not context.strip():
        st.warning("Please provide some context.")
    else:
        with st.spinner("Generating question..."):
            st.session_state.context = context
            st.session_state.submitted = False
            st.session_state.selected_option = None  # Explicitly reset selection
            st.session_state.question_count += 1

            question, options, correct_answer, error = generate_question_and_options(context)
            if error:
                st.error(error)
            else:
                st.session_state.question = question
                st.session_state.options = options
                st.session_state.correct_answer = correct_answer

if st.session_state.question:
    st.subheader("Question:")
    st.write(st.session_state.question)

    st.subheader("Options:")
    # Use a callback to update selected_option only when user interacts
    def update_selection():
        st.session_state.selected_option = st.session_state[f"radio_{st.session_state.question_count}"]

    # Set index=None to prevent pre-selection, only use selected_option if it’s been set
    default_index = st.session_state.options.index(st.session_state.selected_option) if st.session_state.selected_option in st.session_state.options else None
    st.radio(
        "Choose the correct answer:",
        st.session_state.options,
        key=f"radio_{st.session_state.question_count}",
        index=default_index,  # Only pre-select if user has chosen before
        on_change=update_selection  # Update state only on user action
    )

    if st.button("Submit Answer"):
        st.session_state.submitted = True

if st.session_state.submitted and st.session_state.selected_option:
    if st.session_state.selected_option == st.session_state.correct_answer:
        st.success("Correct! 🎉 Well done!")
    else:
        st.error(f"Wrong! The correct answer is: **{st.session_state.correct_answer}**")

st.markdown("---")
st.write("Enter some text, click 'Generate Question' to start, or 'Generate Another Question' to get a new one from the same text. Use 'Reset' to clear everything.")