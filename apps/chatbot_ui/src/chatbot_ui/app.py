import streamlit as st
import html
import uuid

from chatbot_ui.core.config import config
import requests  

st.set_page_config(page_title="Amazon Product Assistant", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
      .context-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.6rem;
        background: #ffffff;
      }
      .context-meta {color: #6b7280; font-size: 0.85rem;}
      .suggestion-card {
        display: flex;
        gap: 0.75rem;
        align-items: center;
      }
      .suggestion-image {
        width: 140px;
        height: 140px;
        object-fit: contain;
        border-radius: 10px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Amazon Product Assistant")
st.caption("Ask about items in our inventory and see the products used.")
st.divider()


def api_call(method, url, **kwargs):
    
    def _show_error_popup(message):
        """Show an error popup with the given message"""
        st.session_state["error_popup"] = {"visible": True,
         "message": message,
        }
    
    try:
        response = getattr(requests, method)(url, **kwargs)
        
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid JSON formt response"}
        
        if response.status_code in (200, 201):
            return True, response_data
        
        return False, response_data
    
    except requests.exceptions.ConnectionError as e:
        _show_error_popup("Connection Error. Please check your internet connection and try again.")
        return False, {"message": f"Connection Error. {str(e)}"}
    except requests.exceptions.Timeout:
        _show_error_popup("Request Timeout. Please try again later.")
        return False, {"message": "Request Timeout."}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": f"An unexpected error occurred {str(e)}."}


def render_used_context(context_items, container):
    container.markdown("**Suggestions**")
    for item in context_items:
        image_url = item.get("image_url")
        product_id = html.escape(str(item.get("id", "")))
        price_value = item.get("price")
        price = html.escape(str(price_value)) if price_value is not None else ""
        meta_html = "<br/>".join(
            line for line in [f"Price: {price}" if price else "", f"Product ID: {product_id}" if product_id else ""] if line
        )
        image_html = (
            f'<img class="suggestion-image" src="{html.escape(image_url)}" alt="product image" />'
            if image_url
            else '<div class="suggestion-image"></div>'
        )
        container.markdown(
            f"""
            <div class="context-card suggestion-card">
              {image_html}
              {f'<div class="context-meta">{meta_html}</div>' if meta_html else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_feedback_url():
    base_url = config.API_URL.rsplit("/", 1)[0]
    return f"{base_url}/feedback"


def submit_feedback(run_id, score, comment=""):
    # print the values of run_id, score, comment
    print(f"run_id: {run_id}, score: {score}, comment: {comment}")
    payload = {
        "run_id": run_id,
        "score": score,
        "comment": comment,
        "source": "api",
    }
    return api_call("post", get_feedback_url(), json=payload)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help you on Amazon Products?"}]
if "latest_context" not in st.session_state:
    st.session_state.latest_context = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "feedback_open" not in st.session_state:
    st.session_state.feedback_open = {}
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            run_id = message.get("run_id")
            if not run_id:
                st.caption("Feedback unavailable for this response.")
            else:
                feedback_entry = st.session_state.feedback.get(run_id)
                if feedback_entry:
                    st.caption("Thanks for your feedback.")
                else:
                    col_up, col_down = st.columns([1, 1])
                    with col_up:
                        if st.button("👍", key=f"feedback_up_{run_id}", disabled=not run_id):
                            result = submit_feedback(run_id, 1)
                            if result[0]:
                                st.session_state.feedback[run_id] = {"score": 1, "comment": ""}
                                st.rerun()
                            else:
                                st.error(result[1].get("message", "Failed to submit feedback."))
                    with col_down:
                        if st.button("👎", key=f"feedback_down_{run_id}", disabled=not run_id):
                            st.session_state.feedback_open[run_id] = True
                    if st.session_state.feedback_open.get(run_id):
                        comment_key = f"feedback_comment_{run_id}"
                        st.text_area(
                            "Reason for thumbs down (optional)",
                            key=comment_key,
                            placeholder="Tell us what was missing or incorrect.",
                        )
                        if st.button("Submit feedback", key=f"feedback_submit_{run_id}"):
                            comment = st.session_state.get(comment_key, "")
                            result = submit_feedback(run_id, 0, comment)
                            if result[0]:
                                st.session_state.feedback[run_id] = {"score": 0, "comment": comment}
                                st.session_state.feedback_open[run_id] = False
                                st.rerun()
                            else:
                                st.error(result[1].get("message", "Failed to submit feedback."))

if prompt := st.chat_input("Hello, how can I help you on Amazon Products?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        output = api_call(
            "post",
            f"{config.API_URL}",
            json={"query": prompt, "thread_id": st.session_state.thread_id},
        )
        if output[0]:
            answer = output[1].get("answer", "")
            used_context = output[1].get("used_context", [])
            run_id = output[1].get("trace_id") if output[1].get("trace_id") else None
            st.write(answer)
            st.session_state.latest_context = used_context
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "run_id": run_id}
            )
            st.rerun()
        else:
            st.write(output[1].get("message", "Request failed."))

with st.sidebar:
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, how can I help you on Amazon Products?"}
        ]
        st.session_state.latest_context = []
        st.rerun()
    if st.session_state.latest_context:
        render_used_context(st.session_state.latest_context, st.sidebar)
    else:
        st.markdown("**Suggestions**")
        st.caption("Ask a question to see related products here.")