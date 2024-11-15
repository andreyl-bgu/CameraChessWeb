import * as tf from "@tensorflow/tfjs-core";
import { getInvTransform, transformBoundary, transformCenters } from "./warp";
import { invalidVideo } from "./detect";
import { detect, getKeypoints, getSquares, getUpdate } from "./findPieces";
import { Chess, Color, Piece, PieceSymbol, Square } from "chess.js";
import { PIECE_SYMBOLS, SQUARE_NAMES } from "./constants";
import { gameResetMoves, gameSetFen, gameSetStart } from "../slices/gameSlice";
import { renderState } from "./render/renderState";
import { SetStringArray } from "../types";

interface findFenInput {
  piecesModelRef: any,
  videoRef: any,
  cornersRef: any,
  canvasRef: any,
  dispatch: any,
  setText: SetStringArray,
  color: Color
}

const getFenAndError = (board: Chess, color: Color) => {
  let fen = board.fen();
  const otherColor: Color = (color === "w") ? "b" : "w";
  fen = fen.replace(` ${otherColor} `, ` ${color} `);

  let error = null;

  // Side to move has opponent in check
  for (let i = 0; i < 64; i++) {
    const square: Square = SQUARE_NAMES[i];
    const piece: Piece = board.get(square);

    const isKing: boolean = (piece.type === "k");
    const isOtherColor: boolean = (piece.color === otherColor);
    const isAttacked: boolean = board.isAttacked(square, color);

    if (isKing && isOtherColor && isAttacked) {
      error = "Side to move has opponent in check";
      return { fen, error }
    }
  }

  return { fen, error }
}

const setFenFromState = (state: number[][], color: Color, dispatch: any, setText: SetStringArray) => {
  const assignment = Array(64).fill(-1);

  // In the first pass, assign the black king
  let bestBlackKingScore = -1;
  let bestBlackKingIdx = -1;
  for (let i = 0; i < 64; i++) {
    const blackKingScore = state[i][1];
    if (blackKingScore > bestBlackKingScore) {
      bestBlackKingScore = blackKingScore;
      bestBlackKingIdx = i;
    }
  }
  assignment[bestBlackKingIdx] = 1;

  // In the second pass, assign the white king
  let bestWhiteKingScore = -1;
  let bestWhiteKingIdx = -1;
  for (let i = 0; i < 64; i++) {
    if (i == bestBlackKingIdx) {
      continue
    }
    const whiteKingScore = state[i][7];
    if (whiteKingScore > bestWhiteKingScore) {
      bestWhiteKingScore = whiteKingScore;
      bestWhiteKingIdx = i;
    }
  }
  assignment[bestWhiteKingIdx] = 7;

  // In the third pass, assign the remaining pieces
  const remainingPieceIdxs = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11];
  for (let i = 0; i < 64; i++) {
    // Square has already been assigned
    if (assignment[i] !== -1) {
      continue
    }

    let bestIdx = null;
    let bestScore = 0.3;
    remainingPieceIdxs.forEach(j => {
      const square: Square = SQUARE_NAMES[i];
      const badRank: boolean = (square[1] === "1") || (square[1] === "8");
      const isPawn: boolean = (PIECE_SYMBOLS[j % 6] === "p");
      if (isPawn && badRank) {
        return;
      }

      const score = state[i][j];
      if (score > bestScore) {
        bestIdx = j;
        bestScore = score;
      }
    });

    if (bestIdx !== null) {
      assignment[i] = bestIdx;
    }
  }

  const board = new Chess();
  board.clear();
  for (let i = 0; i < 64; i++) {
    if (assignment[i] === -1) {
      continue;
    }
    const piece: PieceSymbol = PIECE_SYMBOLS[assignment[i] % 6];
    const color: Color = (assignment[i] > 5) ? 'w' : 'b';
    const square: Square = SQUARE_NAMES[i];
    board.put({ 'type': piece, 'color': color }, square);
  }

  const { fen, error } = getFenAndError(board, color);
  if (error === null) {
    dispatch(gameSetStart(fen));
    dispatch(gameSetFen(fen));
    dispatch(gameResetMoves());
    setText(["Set starting FEN"]);
  } else {
    setText(["Invalid FEN:", error]);
  }
}

export const _findMultipleFen = async ({ piecesModelRef, videoRef, cornersRef, canvasRef, dispatch, setText, color }: findFenInput) => {
  if (invalidVideo(videoRef)) {
    return;
  }
  const keypoints1: number[][] = getKeypoints(cornersRef, canvasRef);
  const invTransform1 = getInvTransform(keypoints1);
  const [centers1, centers3D1] = transformCenters(invTransform1);
  const [boundary1, boundary3D1] = transformBoundary(invTransform1);
  const { boxes: boxes1, scores: scores1 } = await detect(piecesModelRef, videoRef, keypoints1);
  const squares1: number[] = getSquares(boxes1, centers3D1, boundary3D1);
  const state1 = getUpdate(scores1, squares1);
  setFenFromState(state1, color, dispatch, setText);

  const keypoints2: number[][] = getKeypoints(cornersRef, canvasRef);
  const invTransform2 = getInvTransform(keypoints2);
  const [centers2, centers3D2] = transformCenters(invTransform2);
  const [boundary2, boundary3D2] = transformBoundary(invTransform2);
  const { boxes: boxes2, scores: scores2 } = await detect(piecesModelRef, videoRef, keypoints2);
  const squares2: number[] = getSquares(boxes2, centers3D2, boundary3D2);
  const state2 = getUpdate(scores2, squares2);
  setFenFromState(state2, color, dispatch, setText);

  renderState(canvasRef.current, centers1, boundary1, state1);
  renderState(canvasRef.current, centers2, boundary2, state2);

  tf.dispose([boxes1, scores1, centers3D1, boundary3D1, boxes2, scores2, centers3D2, boundary3D2]);
}

export const findMultipleFen = async ({ piecesModelRef, videoRef, cornersRef, canvasRef, dispatch, setText, color }: findFenInput) => {
  const startTensors = tf.memory().numTensors;

  await _findMultipleFen({ piecesModelRef, videoRef, cornersRef, canvasRef, dispatch, setText, color });

  const endTensors = tf.memory().numTensors;
  if (startTensors < endTensors) {
    console.error(`Memory Leak! (${endTensors} > ${startTensors})`)
  }

  return () => {
    tf.disposeVariables();
  };
}