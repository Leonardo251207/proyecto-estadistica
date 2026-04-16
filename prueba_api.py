import google.generativeai as genai

# Asegúrate de que esta clave comience con AIza
api_key = "TU_API_KEY_AQUI" 

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hola, ¿puedes saludarme?")
    print("✅ ¡ÉXITO! Respuesta:", response.text)
except Exception as e:
    print(f"❌ ERROR: {e}")