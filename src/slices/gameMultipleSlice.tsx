import { createSlice } from '@reduxjs/toolkit';
import { useSelector } from 'react-redux';
import { Game, RootState } from '../typesMultiple';
import { START_FEN } from '../utils/constants';
import { Chess } from 'chess.js';

const initialState: { board1: Game, board2: Game } = {
  board1: {
    moves: "",
    fen: START_FEN,
    start: START_FEN,
    lastMove: "",
    greedy: false
  },
  board2: {
    moves: "",
    fen: START_FEN,
    start: START_FEN,
    lastMove: "",
    greedy: false
  }
};

type BoardPayload = { board: 'board1' | 'board2', moves?: string, fen?: string, start?: string, lastMove?: string, greedy?: boolean };

const gameMultipleSlice = createSlice({
  name: 'gameMultiple',
  initialState,
  reducers: {
    gameSetMoves(state, action: { payload: BoardPayload }) {
      state[action.payload.board].moves = action.payload.moves!;
    },
    gameSetFen(state, action: { payload: BoardPayload }) {
      state[action.payload.board].fen = action.payload.fen!;
    },
    gameSetStart(state, action: { payload: BoardPayload }) {
      state[action.payload.board].start = action.payload.start!;
    },
    gameSetLastMove(state, action: { payload: BoardPayload }) {
      state[action.payload.board].lastMove = action.payload.lastMove!;
    },
    gameResetMoves(state, action: { payload: BoardPayload }) {
      state[action.payload.board].moves = initialState[action.payload.board].moves;
    },
    gameResetFen(state, action: { payload: BoardPayload }) {
      state[action.payload.board].fen = initialState[action.payload.board].fen;
    },
    gameResetStart(state, action: { payload: BoardPayload }) {
      state[action.payload.board].start = initialState[action.payload.board].start;
    },
    gameResetLastMove(state, action: { payload: BoardPayload }) {
      state[action.payload.board].lastMove = initialState[action.payload.board].lastMove;
    },
    gameUpdate(state, action: { payload: BoardPayload }) {
      const newState: Game = {
        moves: action.payload.moves!,
        fen: action.payload.fen!,
        start: action.payload.start!,
        lastMove: action.payload.lastMove!,
        greedy: action.payload.greedy!
      };
      state[action.payload.board] = newState;
    }
  }
});

const getMovesFromPgn = (board: Chess) => {
  const pgn = board.pgn();
  const moves = pgn.replace(/\[.*?\]/g, '').replace(/\r?\n|\r/g, '');
  return moves;
};

export const gameMultipleSelect = () => {
  return useSelector((state: RootState) => state.gameMultiple);
};

export const makePgn = (game: Game) => {
  return `[FEN "${game.start}"]` + "\n \n" + game.moves;
};

export const makeUpdatePayload = (board: Chess, greedy: boolean = false) => {
  const history = board.history({ verbose: true });
  const moves = getMovesFromPgn(board);
  const fen = board.fen();
  const lastMove = (history.length === 0) ? "" : history[history.length - 1].lan;

  const payload = {
    moves,
    fen,
    lastMove,
    greedy
  };

  return payload;
};

export const makeBoard = (game: Game): Chess => {
  const board = new Chess(game.start);
  board.loadPgn(makePgn(game));
  return board;
};

export const {
  gameSetMoves, gameResetMoves,
  gameSetFen, gameResetFen,
  gameSetStart, gameResetStart,
  gameSetLastMove, gameResetLastMove, gameUpdate
} = gameMultipleSlice.actions;
export default gameMultipleSlice.reducer;