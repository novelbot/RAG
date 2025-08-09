import sys
import traceback
from src.core.config import get_config, LLMConfig
from src.llm.manager import create_llm_manager
from src.llm.base import ProviderConfig, LLMProvider

try:
    # 1. 기본 설정 확인
    config = get_config()
    print(f"Default config.llm has stream: {hasattr(config.llm, 'stream')}")
    if hasattr(config.llm, 'stream'):
        print(f"Default config.llm.stream = {config.llm.stream}")
    
    # 2. 새 LLMConfig 생성
    new_config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=config.llm.api_key,  # Use API key from env
        temperature=0.3,
        max_tokens=1000
    )
    print(f"\nNew LLMConfig has stream: {hasattr(new_config, 'stream')}")
    if hasattr(new_config, 'stream'):
        print(f"New LLMConfig.stream = {new_config.stream}")
    
    # 3. ProviderConfig 생성
    provider_config = ProviderConfig(
        provider=LLMProvider.OPENAI,
        config=new_config
    )
    
    # 4. LLM Manager 생성 시도
    print("\nTrying to create LLM manager...")
    try:
        llm_manager = create_llm_manager([provider_config])
        print("✅ LLM manager created successfully")
    except Exception as e:
        print(f"❌ Failed to create LLM manager: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
