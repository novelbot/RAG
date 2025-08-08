import asyncio
from src.conversation.storage import conversation_storage

async def test_conversation_retrieval():
    conversation_id = "58999a96-6567-4fa5-9434-2b945c098696"
    
    # Test direct retrieval
    print(f"Testing conversation ID: {conversation_id}")
    
    # Get conversation info
    conv_info = await conversation_storage.get_conversation_info(conversation_id)
    print(f"\nConversation Info: {conv_info}")
    
    # Get messages
    messages = await conversation_storage.get_messages(conversation_id, limit=20)
    print(f"\nNumber of messages retrieved: {len(messages)}")
    
    if messages:
        print("\nMessages:")
        for msg in messages:
            print(f"  - {msg.role}: {msg.content[:50]}...")
    else:
        print("No messages found\!")
    
    # Check the actual user_id
    if conv_info:
        print(f"\nConversation user_id: {conv_info['user_id']}")

if __name__ == "__main__":
    asyncio.run(test_conversation_retrieval())
