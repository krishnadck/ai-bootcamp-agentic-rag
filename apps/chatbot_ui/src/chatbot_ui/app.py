import streamlit as st

from chatbot_ui.core.config import config
import requests  


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
        _show_error_popup(f"Connection Error. Please check your internet connection and try again.")
        return False, {"message": f"Connection Error. {str(e)}"}
    except requests.exceptions.Timeout:
        _show_error_popup("Request Timeout. Please try again later.")
        return False, {"message": "Request Timeout."}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": f"An unexpected error occurred {str(e)}."}


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, how can I help you on Amazon Products?"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello, how can I help you on Amazon Products?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        output = api_call("post", f"{config.API_URL}", 
                          json={"query": prompt})
        #st.write(output)
        response_data = output[1]["response"]
        answer = response_data["answer"]
        st.write(answer)
        if output[0]:
            st.session_state.messages.append({"role": "assistant", "content": answer})