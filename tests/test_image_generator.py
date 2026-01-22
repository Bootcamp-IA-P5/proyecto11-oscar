import pytest
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image
from src.models.image_generator import (
    translate_to_english_keywords,
    extract_keywords,
    create_placeholder_image,
    generate_image_from_huggingface,
    generate_image_from_replicate,
    search_image_from_unsplash,
    search_image_from_pexels
)

def test_translate_to_english_keywords():
    assert translate_to_english_keywords("agujeros negros") == "black hole space galaxy"
    assert translate_to_english_keywords("inteligencia artificial") == "artificial intelligence technology"
    assert translate_to_english_keywords("universo") == "universo" # Fallback cleaned

def test_extract_keywords():
    prompt = "A photorealistic image of a modern drone hovering above a futuristic city at sunset, 8k resolution"
    keywords = extract_keywords(prompt)
    assert "drone" in keywords
    assert "hovering" in keywords # Should be in top 3
    assert "8k" not in keywords # Should be filtered

def test_create_placeholder_image():
    img = create_placeholder_image("test")
    assert isinstance(img, Image.Image)
    assert img.size == (640, 480)

@patch("src.models.image_generator.InferenceClient")
def test_generate_image_from_huggingface_success(mock_client):
    mock_instance = mock_client.return_value
    mock_img = Image.new('RGB', (1024, 1024))
    mock_instance.text_to_image.return_value = mock_img
    
    result = generate_image_from_huggingface("test prompt")
    
    assert result == mock_img
    mock_instance.text_to_image.assert_called_once()

@patch("src.models.image_generator.InferenceClient")
def test_generate_image_from_huggingface_fallback(mock_client):
    mock_instance = mock_client.return_value
    mock_instance.text_to_image.side_effect = Exception("Rate limit reached")
    
    result = generate_image_from_huggingface("test prompt")
    
    assert isinstance(result, Image.Image) # Should return placeholder
    assert result.size == (640, 480)

@patch("src.models.image_generator.replicate.run")
@patch("src.models.image_generator.requests.get")
@patch("builtins.open", new_callable=mock_open)
def test_generate_image_from_replicate_success(mock_file, mock_get, mock_replicate):
    mock_replicate.return_value = ["http://fakeurl.com/image.webp"]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake image content"
    mock_get.return_value = mock_response
    
    result = generate_image_from_replicate("test prompt")
    
    assert result == "generated_content.webp"
    mock_file().write.assert_called_once_with(b"fake image content")

@patch("src.models.image_generator.requests.get")
@patch("src.models.image_generator.UNSPLASH_ACCESS_KEY", "fake_key")
def test_search_image_from_unsplash_success(mock_get):
    # Mock API call
    mock_api_resp = MagicMock()
    mock_api_resp.status_code = 200
    mock_api_resp.json.return_value = {"urls": {"regular": "http://unsplash.com/img.jpg"}, "user": {"name": "Photographer"}}
    
    # Mock Image download
    mock_img_resp = MagicMock()
    mock_img_resp.status_code = 200
    mock_img_resp.content = b"fake image content"
    
    mock_get.side_effect = [mock_api_resp, mock_img_resp]
    
    with patch("src.models.image_generator.Image.open") as mock_open_img:
        search_image_from_unsplash("space")
        mock_open_img.assert_called_once()

@patch("src.models.image_generator.requests.get")
@patch("src.models.image_generator.PEXELS_API_KEY", "fake_key")
def test_search_image_from_pexels_success(mock_get):
    # Mock API call
    mock_api_resp = MagicMock()
    mock_api_resp.status_code = 200
    mock_api_resp.json.return_value = {"photos": [{"src": {"large2x": "http://pexels.com/img.jpg"}, "photographer": "Artist"}]}
    
    # Mock Image download
    mock_img_resp = MagicMock()
    mock_img_resp.status_code = 200
    mock_img_resp.content = b"fake image content"
    
    mock_get.side_effect = [mock_api_resp, mock_img_resp]
    
    with patch("src.models.image_generator.Image.open") as mock_open_img:
        search_image_from_pexels("science")
        mock_open_img.assert_called_once()
