from langchain.memory import ConversationBufferWindowMemory

user_memory = {}

def get_user_memory(user_id: str, k: int = 7):
    if user_id not in user_memory:
        user_memory[user_id] = ConversationBufferWindowMemory(
            k=k,
            return_messages=True
        )
    return user_memory[user_id]