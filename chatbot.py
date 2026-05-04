print("AI Chatbot (type 'exit' to quit)")

while True:
    user = input("You: ").lower()

    if user == "exit":
        print("Bot: Goodbye!")
        break

    elif "hello" in user or "hi" in user:
        print("Bot: Hello! How can I help you?")

    elif "name" in user:
        print("Bot: I am a simple AI chatbot.")

    elif "how are you" in user:
        print("Bot: I'm doing great!")

    elif "bye" in user:
        print("Bot: Bye! Have a nice day!")

    else:
        print("Bot: Sorry, I don't understand.")
