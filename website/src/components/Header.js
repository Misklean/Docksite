// Header.js
import React from 'react';
import { useNavigate } from 'react-router-dom';

const Header = () => {
    const navigate = useNavigate();

    const navigateToHome = () => {
	// Navigate to '/tictactoe'
	navigate('/');
    };
    const navigateToTictactoe = () => {
	// Navigate to '/tictactoe'
	navigate('/tictactoe');
    };
    const navigateToPygame = () => {
	// Navigate to '/tictactoe'
	navigate('/pygame');
    };
    return (
	<header>
	    <div>
		<hr />
		<button onClick={navigateToHome}>HomePage</button>
		<button onClick={navigateToTictactoe}>TicTacToe</button>
		<button onClick={navigateToPygame}>PyGame</button>
            </div>
	</header>
    );
};

export default Header;
