import { useState, useEffect } from "react";
import { loadPyodide } from "pyodide";
import script from './main.py';

const runScript = async (code) => {
	const pyodide = await loadPyodide({
	  indexURL : "https://cdn.jsdelivr.net/pyodide/v0.23.4/full"
	});
  
	pyodide.loadPackage('https://pypi.org/project/pyboy/');
	return await pyodide.runPythonAsync(code);
  }
  
const PyGame = () => {
	const [output, setOutput] = useState("(loading...)");

	useEffect(() => {
	  const run = async () => {
		const scriptText = await (await fetch(script)).text();
		const out = await runScript(scriptText);
		setOutput(out);
	  }
	  run();
  
	}, []);
  
	return (
	  <div className="App">
		  <p>
			{output}
		  </p>
	  </div>
	);
  
  }

//     const [data, setData] = useState(null);
//     const [isLoading, setIsLoading] = useState(false);
//     const [err, setErr] = useState('');

//     const handleGetGameImage = async () => {
// 	setIsLoading(true);

// 	try {
// 	    const response = await fetch('http://127.0.0.1:5000/pygame', {
// 		method: 'GET',
// 		headers: {
// 		    "Content-Type": 'application/json',
// 		}
// 	    });

// 	    if (!response.ok) {
// 		console.log('not good');
// 		throw new Error(`Error! status: ${response.status}`);
// 	    }

// 	    console.log("WE DID IT");
	    
// 	    const byteData = await response.arrayBuffer();
// 	    const imageData = btoa(String.fromCharCode(...new Uint8Array(byteData)));

// 	    setData(imageData);
// 	} catch (err) {
// 	    setErr(err.message);
// 	} finally {
// 	    setIsLoading(false);
// 	}

//     };
    
//     const handlePostActionLeftArrow = async () => {
// 	setIsLoading(true);

// 	try {
// 	    const response = await fetch('http://127.0.0.1:5000/pygame/action/leftarrow', {
// 		method: 'POST',
// 		headers: {
// 		    "Content-Type": 'application/json',
// 		},
// 		body: JSON.stringify({})
// 	    });

// 	    if (!response.ok) {
// 		console.log('not good');
// 		throw new Error(`Error! status: ${response.status}`);
// 	    }

// 	    const result = await response.json();

// 	    console.log('result is: ', JSON.stringify(result, null, 4));

// 	    setData(result);
// 	} catch (err) {
// 	    setErr(err.message);
// 	} finally {
// 	    setIsLoading(false);
// 	}
//     };

//     const handlePostActionRightArrow = async () => {
// 	setIsLoading(true);

// 	try {
// 	    const response = await fetch('http://127.0.0.1:5000/pygame/action/rightarrow', {
// 		method: 'POST',
// 		headers: {
// 		    "Content-Type": 'application/json',
// 		},
// 		body: JSON.stringify({})
// 	    });

// 	    if (!response.ok) {
// 		console.log('not good');
// 		throw new Error(`Error! status: ${response.status}`);
// 	    }

// 	    const result = await response.json();

// 	    console.log('result is: ', JSON.stringify(result, null, 4));

// 	    setData(result);
// 	} catch (err) {
// 	    setErr(err.message);
// 	} finally {
// 	    setIsLoading(false);
// 	}
//     };

//     const handlePostActionUpArrow = async () => {
// 	setIsLoading(true);

// 	try {
// 	    const response = await fetch('http://127.0.0.1:5000/pygame/action/uparrow', {
// 		method: 'POST',
// 		headers: {
// 		    "Content-Type": 'application/json',
// 		},
// 		body: JSON.stringify({})
// 	    });

// 	    if (!response.ok) {
// 		console.log('not good');
// 		throw new Error(`Error! status: ${response.status}`);
// 	    }

// 	    const result = await response.json();

// 	    console.log('result is: ', JSON.stringify(result, null, 4));

// 	    setData(result);
// 	} catch (err) {
// 	    setErr(err.message);
// 	} finally {
// 	    setIsLoading(false);
// 	}
//     };

//     const handlePostActionDownArrow = async () => {
// 	setIsLoading(true);

// 	try {
// 	    const response = await fetch('http://127.0.0.1:5000/pygame/action/downarrow', {
// 		method: 'POST',
// 		headers: {
// 		    "Content-Type": 'application/json',
// 		},
// 		body: JSON.stringify({})
// 	    });

// 	    if (!response.ok) {
// 		console.log('not good');
// 		throw new Error(`Error! status: ${response.status}`);
// 	    }

// 	    const result = await response.json();

// 	    console.log('result is: ', JSON.stringify(result, null, 4));

// 	    setData(result);
// 	} catch (err) {
// 	    setErr(err.message);
// 	} finally {
// 	    setIsLoading(false);
// 	}
//     };

//     useEffect(() => {
// 	const keyDownHandler = event => {
// 	    console.log('User pressed: ', event.key);

// 	    if (event.key === 'ArrowLeft') {
// 		event.preventDefault();
// 		handlePostActionLeftArrow();
// 	    }
// 	    else if (event.key === 'ArrowRight') {
// 		event.preventDefault();
// 		handlePostActionRightArrow();
// 	    }
// 	    else if (event.key === 'ArrowUp') {
// 		event.preventDefault();
// 		handlePostActionUpArrow();
// 	    }
// 	    else if (event.key === 'ArrowDown') {
// 		event.preventDefault();
// 		handlePostActionDownArrow();
// 	    }
// 	};

// 	document.addEventListener('keydown', keyDownHandler);

// 	return () => {
// 	    document.removeEventListener('keydown', keyDownHandler);
// 	};
//     }, []);

//     const gameLoop = () => {
//     handleGetGameImage();
//     window.requestAnimationFrame(gameLoop);
//   };

//   useEffect(() => {
//     gameLoop();
//     return () => window.cancelAnimationFrame(gameLoop);
//   }, []);


// return (
//   <div>
//     <img src={`data:image/png;base64,${data}`} alt=""/>
//   </div>
// );

export default PyGame;
