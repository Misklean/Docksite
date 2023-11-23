import React, { useState, useEffect } from 'react';

const PyGame = () => {
const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/countries')
      .then(response => response.json())
      .then(jsonData => {
        setData(jsonData);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, []);


  return (
    <div>
      {data && (
        <pre>
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
};

export default PyGame;
