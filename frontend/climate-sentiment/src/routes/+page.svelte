<script>
    let text = '';
    let prediction = null;
    let confidence = null; 
  
    async function predict() {
      try {
        const response = await fetch('https://climate-sentiment.onrender.com/predict', { //change to http://localhost:8000/predict if cloned locally
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text }),
        });
  
        if (response.ok) {
          const data = await response.json();
          prediction = data.prediction; 
          confidence = data.confidence; 
        } else {
          console.error('Error in prediction:', response.statusText);
        }
      } catch (error) {
        console.error('Network error:', error);
      }
    }
  </script>
  
  <main>
    <div class="container">
      <h1>Climate Sentiment Analysis</h1>
  
      <p class="description">
        Enter any text below and we'll classify it using a machine learning model trained for text prediction.
      </p>
  
      <input
        type="text"
        bind:value={text}
        placeholder="Enter text here..."
      />
      <button on:click={predict}>Predict</button>
  
      {#if prediction !== null && confidence !== null}
        <p class="result">
          Prediction: {prediction} <br />
          Confidence: {confidence.toFixed(2)}% 
        </p>
      {/if}
    </div>
  </main>
  
  <style>
    :global(html, body) {
      height: 100%;
      width: 100%;
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      overflow: hidden;
    }
  
    main {
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #fff;
    }
  
    .container {
      background-color: rgba(0, 0, 0, 0.3);
      padding: 3rem;
      border-radius: 16px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 90%;
      max-width: 500px;
    }
  
    h1 {
      margin-bottom: 1.5rem;
      font-size: 2rem;
      font-weight: bold;
      text-align: center;
      background: linear-gradient(to right, #8a2be2, #00bfff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
  
    input {
      width: 100%;
      padding: 0.75rem 1rem;
      font-size: 1rem;
      border: none;
      border-radius: 8px;
      margin-bottom: 1rem;
      background-color: rgba(255, 255, 255, 0.1);
      color: white;
      outline: none;
    }
  
    input::placeholder {
      color: #ccc;
    }
  
    button {
      padding: 0.75rem 2rem;
      background: linear-gradient(to right, #8a2be2, #00bfff);
      border: none;
      outline: none;
      border-radius: 8px;
      font-size: 1rem;
      color: white;
      cursor: pointer;
      transition: all 0.3s ease;
    }
  
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(0, 191, 255, 0.4);
    }
  
    .result {
      margin-top: 2rem;
      font-size: 1.2rem;
      color: #66ffcc;
    }
  
    .description {
      font-size: 1rem;
      margin: 0 0 1.5rem 0; 
      text-align: center;
      color: #ddd;
      max-width: 400px;
    }
  </style>
  