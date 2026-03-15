// TicTacToe Game - test_tictactoe.js
// WARNING: Contains intentional structural errors for TST testing

const board = ['', '', '', '', '', '', '', '', ''];
let currentPlayer = 'X'
let gameActive = true;

// STRUCTURAL ERROR 1: Missing closing brace on function
function checkWinner(board) {
    const winPatterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ];

    for (let pattern of winPatterns) {
        const [a, b, c] = pattern;
        if (board[a] && board[a] === board[b] && board[a] === board[c]) {
            return board[a];
        }
    }
    return null;
// missing closing brace here

// STRUCTURAL ERROR 2: Function called before definition + wrong parameter count
function handleMove(index) {
    if (!gameActive || board[index] !== '') return;

    board[index] = currentPlayer;

    const winner = checkWinner(board, currentPlayer);  // extra argument

    if (winner) {
        console.log(`Player ${winner} wins!`);
        gameActive = false;
        return;
    }

    if (!board.includes('')) {
        console.log('Draw!');
        gameActive = false;
        return;
    }

    switchPlayer()  // called before defined, missing semicolon
}

// STRUCTURAL ERROR 3: Variable shadowing + wrong type comparison
function switchPlayer() {
    let currentPlayer = (currentPlayer === 'X') ? 'O' : 'X';  // shadows outer variable
    console.log(`Current player: ${currentPlayer}`);
}

// STRUCTURAL ERROR 4: Missing return statement path
function isBoardFull(board) {
    for (let i = 0; i < board.length; i++) {
        if (board[i] === '') {
            return false;
        }
        // no return true at end
    }
}

function resetGame() {
    board = ['', '', '', '', '', '', '', '', ''];  // STRUCTURAL ERROR 5: const reassignment
    currentPlayer = 'X';
    gameActive = true;
    console.log('Game reset.');
}

// Entry point
handleMove(0);
handleMove(1);
handleMove(3);
handleMove(4);
handleMove(6);  // X wins diagonal
