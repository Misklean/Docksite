import { createRoot } from "react-dom/client";
import { BrowserRouter } from 'react-router-dom';
import "./index.css";

import App from "./App";
// import TicTacToe from "./TicTacToe";

const root = createRoot(document.getElementById("root"));
root.render(
  <BrowserRouter>
    <App />
  </BrowserRouter>
);
