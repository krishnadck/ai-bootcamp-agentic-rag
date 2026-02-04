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
        
        if response.status_code == 200:
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


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help you on Amazon Products?"}]
if "latest_context" not in st.session_state:
    st.session_state.latest_context = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            st.write(answer)
            st.session_state.latest_context = used_context
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
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