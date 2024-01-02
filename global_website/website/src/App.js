import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';

import TicTacToe from './TicTacToe';
import HomePage from './HomePage';
import PyGame from './PyGame';
const App = () => {

    return (
	<>
	<Header />
	    <div>
		<Routes>
		    <Route path="/" element={<HomePage />} />
		    <Route path="/tictactoe" element={<TicTacToe />} />
		    <Route path="/pygame" element={<PyGame />} />
		</Routes>
	    </div>
	</>
    );
};

export default App;
