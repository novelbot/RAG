import sys
import traceback
from src.core.config import get_config, LLMConfig
from src.llm.manager import create_llm_manager, ProviderConfig
from src.llm.base import LLMProvider

try:
    # 1. 기본 설정 확인
    config = get_config()
    print(f"Default config.llm type: {type(config.llm)}")
    print(f"Default config.llm attributes: {[attr for attr in dir(config.llm) if not attr.startswith('_')][:10]}")
    
    # 2. 새 LLMConfig 생성
    new_config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=config.llm.api_key,
        temperature=0.7,
        max_tokens=1000
    )
    print(f"\nNew LLMConfig type: {type(new_config)}")
    print(f"New LLMConfig.stream exists: {hasattr(new_config, 'stream')}")
    
    # 3. ProviderConfig 생성
    provider_config = ProviderConfig(
        provider=LLMProvider.OPENAI,
        config=new_config
    )
    
    print(f"\nProviderConfig.config type: {type(provider_config.config)}")
    print(f"ProviderConfig.config.stream exists: {hasattr(provider_config.config, 'stream')}")
    
    # 4. LLM Manager 생성 시도
    print("\nTrying to create LLM manager...")
    try:
        llm_manager = create_llm_manager([provider_config])
        print("✅ LLM manager created successfully")
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Other error: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
