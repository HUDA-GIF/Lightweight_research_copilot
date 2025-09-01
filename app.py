import streamlit as st
import time

# --- Initialize session state ---
if "proceed_clicked" not in st.session_state:
    st.session_state.proceed_clicked = False

# --- App Title ---
st.title("📚 Lightweight Research Copilot")

# --- Text Input ---
text = st.text_area("Paste your text here:")

# --- Button to process text ---
if st.button("🔍 Proceed"):
    if text.strip():
        st.session_state.proceed_clicked = True
        with st.spinner("Processing text..."):
            time.sleep(1.5)  # Simulate processing delay
            st.session_state.summary = (
                "Artificial intelligence (AI) is a branch of computer science that focuses "
                "on building systems capable of performing tasks that typically require human "
                "intelligence. AI is used in healthcare for diagnostics, in finance for fraud "
                "detection, and in many industries to improve efficiency. Machine learning, "
                "a subfield of AI, involves training algorithms on large datasets to make "
                "predictions or decisions without being explicitly programmed."
            )
    else:
        st.warning("⚠️ Please paste some text before proceeding.")

# --- Show summary & quiz if Proceed was clicked ---
if st.session_state.proceed_clicked:
    st.subheader("📝 Summary")
    st.write(st.session_state.summary)

    st.subheader("📌 Quiz")
    q1 = st.text_input("1. What is the main topic of the text?", key="q1")
    q2 = st.text_input("2. Mention one important detail from the summary.", key="q2")
    q3 = st.text_input("3. Explain one challenge or application discussed.", key="q3")

    if q1 or q2 or q3:
        st.subheader("✅ Results")

        # Check Q1
        if q1:
            if "artificial intelligence" in q1.lower():
                st.success("Q1: Correct!")
            else:
                st.error("Q1: Incorrect. Hint: It's about AI.")

        # Check Q2
        if q2:
            if "healthcare" in q2.lower() or "finance" in q2.lower():
                st.success("Q2: Correct!")
            else:
                st.error("Q2: Incorrect. Hint: Mention an industry.")

        # Check Q3
        if q3:
            if any(word in q3.lower() for word in ["bias", "privacy", "job", "healthcare", "finance"]):
                st.success("Q3: Correct!")
            else:
                st.error("Q3: Incorrect. Hint: Talk about a challenge or application.")
