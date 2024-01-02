import { useState, useEffect } from "react";

  
const PyGame = () => {
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [err, setErr] = useState('');

    useEffect(() => {
    const gameLoop = () => {
      handleGetGameImage();
      const timeout = setTimeout(gameLoop, 1000 / 60);
      return () => clearTimeout(timeout);
    };

    window.requestAnimationFrame(gameLoop);
  }, []);

  const handleGetGameImage = async () => {
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/pygame', {
        method: 'GET',
        headers: {
          "Content-Type": 'application/json',
        },
      });

      if (!response.ok) {
        console.log('not good');
        throw new Error(`Error! status: ${response.status}`);
      }

      const byteData = await response.arrayBuffer();
      const imageData = btoa(String.fromCharCode(...new Uint8Array(byteData)));

      setData(imageData);
    } catch (err) {
      setErr(err.message);
    } finally {
      setIsLoading(false);
    }
  };

    
    const handlePostActionLeftArrow = async () => {
	setIsLoading(true);

	try {
	    const response = await fetch('http://127.0.0.1:5000/pygame/action/leftarrow', {
		method: 'POST',
		headers: {
		    "Content-Type": 'application/json',
		},
		body: JSON.stringify({})
	    });

	    if (!response.ok) {
		console.log('not good');
		throw new Error(`Error! status: ${response.status}`);
	    }

	    const result = await response.json();

	    console.log('result is: ', JSON.stringify(result, null, 4));

	    setData(result);
	} catch (err) {
	    setErr(err.message);
	} finally {
	    setIsLoading(false);
	}
    };

    const handlePostActionRightArrow = async () => {
	setIsLoading(true);

	try {
	    const response = await fetch('http://127.0.0.1:5000/pygame/action/rightarrow', {
		method: 'POST',
		headers: {
		    "Content-Type": 'application/json',
		},
		body: JSON.stringify({})
	    });

	    if (!response.ok) {
		console.log('not good');
		throw new Error(`Error! status: ${response.status}`);
	    }

	    const result = await response.json();

	    console.log('result is: ', JSON.stringify(result, null, 4));

	    setData(result);
	} catch (err) {
	    setErr(err.message);
	} finally {
	    setIsLoading(false);
	}
    };

    const handlePostActionUpArrow = async () => {
	setIsLoading(true);

	try {
	    const response = await fetch('http://127.0.0.1:5000/pygame/action/uparrow', {
		method: 'POST',
		headers: {
		    "Content-Type": 'application/json',
		},
		body: JSON.stringify({})
	    });

	    if (!response.ok) {
		console.log('not good');
		throw new Error(`Error! status: ${response.status}`);
	    }

	    const result = await response.json();

	    console.log('result is: ', JSON.stringify(result, null, 4));

	    setData(result);
	} catch (err) {
	    setErr(err.message);
	} finally {
	    setIsLoading(false);
	}
    };

    const handlePostActionDownArrow = async () => {
	setIsLoading(true);

	try {
	    const response = await fetch('http://127.0.0.1:5000/pygame/action/downarrow', {
		method: 'POST',
		headers: {
		    "Content-Type": 'application/json',
		},
		body: JSON.stringify({})
	    });

	    if (!response.ok) {
		console.log('not good');
		throw new Error(`Error! status: ${response.status}`);
	    }

	    const result = await response.json();

	    console.log('result is: ', JSON.stringify(result, null, 4));

	    setData(result);
	} catch (err) {
	    setErr(err.message);
	} finally {
	    setIsLoading(false);
	}
    };

        const handlePostActionButtonA = async () => {
	setIsLoading(true);

	try {
	    const response = await fetch('http://127.0.0.1:5000/pygame/action/buttona', {
		method: 'POST',
		headers: {
		    "Content-Type": 'application/json',
		},
		body: JSON.stringify({})
	    });

	    if (!response.ok) {
		console.log('not good');
		throw new Error(`Error! status: ${response.status}`);
	    }

	    const result = await response.json();

	    console.log('result is: ', JSON.stringify(result, null, 4));

	    setData(result);
	} catch (err) {
	    setErr(err.message);
	} finally {
	    setIsLoading(false);
	}
	};

            const handlePostActionButtonB = async () => {
	setIsLoading(true);

	try {
	    const response = await fetch('http://127.0.0.1:5000/pygame/action/buttonb', {
		method: 'POST',
		headers: {
		    "Content-Type": 'application/json',
		},
		body: JSON.stringify({})
	    });

	    if (!response.ok) {
		console.log('not good');
		throw new Error(`Error! status: ${response.status}`);
	    }

	    const result = await response.json();

	    console.log('result is: ', JSON.stringify(result, null, 4));

	    setData(result);
	} catch (err) {
	    setErr(err.message);
	} finally {
	    setIsLoading(false);
	}
    };

    useEffect(() => {
	const keyDownHandler = event => {
	    console.log('User pressed: ', event.key);

	    if (event.key === 'ArrowLeft') {
		event.preventDefault();
		handlePostActionLeftArrow();
	    }
	    else if (event.key === 'ArrowRight') {
		event.preventDefault();
		handlePostActionRightArrow();
	    }
	    else if (event.key === 'ArrowUp') {
		event.preventDefault();
		handlePostActionUpArrow();
	    }
	    else if (event.key === 'ArrowDown') {
		event.preventDefault();
		handlePostActionDownArrow();
	    }
	    else if (event.key === 'a') {
		event.preventDefault();
		handlePostActionButtonA();
	    }
	    else if (event.key === 'b') {
		event.preventDefault();
		handlePostActionButtonB();
	    }
	};

	document.addEventListener('keydown', keyDownHandler);

	return () => {
	    document.removeEventListener('keydown', keyDownHandler);
	};
    }, []);

	return (
	<div>
		<img src={`data:image/png;base64,${data}`} alt=""/>
	</div>
	);

}

export default PyGame;
