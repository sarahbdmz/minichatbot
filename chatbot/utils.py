class ConversationManager:
    def __init__(self, max_context=5):
        self.conversation_history = []
        self.max_context = max_context

    def add_interaction(self, user_msg, bot_msg):
        self.conversation_history.append(('user', user_msg))
        self.conversation_history.append(('assistant', bot_msg))
        if len(self.conversation_history) > self.max_context * 2:
            self.conversation_history = self.conversation_history[-self.max_context*2:]

    def get_context(self):
        context = ""
        for speaker, msg in self.conversation_history[-self.max_context*2:]:
            prefix = "User: " if speaker == 'user' else "Assistant: "
            context += prefix + msg + " "
        return context.strip()

    def clear_history(self):
        self.conversation_history = []
