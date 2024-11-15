import { renderMultipleState } from "./render/renderMultipleState";
import * as tf from "@tensorflow/tfjs-core";
import { getInvTransform, transformBoundary, transformCenters } from "./warp";
import { gameUpdate, makeUpdatePayload } from "../slices/gameSlice";
import { getBoxesAndScores, getInput } from "./detectMultiple";
import { Mode, MovesData, MovesPair } from "../types";
import { zeros } from "./math";
import {CORNER_KEYS, MARKER_DIAMETER, MODEL_HEIGHT, MODEL_WIDTH} from "./constants";
import { Chess } from "chess.js";

const calculateScore = (state: any, move: MovesData, from_thr=0.6, to_thr=0.6) => {
  let score = 0;
  move.from.forEach(square => {
    score += 1 - Math.max(...state[square]) - from_thr;
  });

  for (let i = 0; i < move.to.length; i++) {
    score += state[move.to[i]][move.targets[i]] - to_thr;
  }

  return score;
};

const processState = (state: any, movesPairs: MovesPair[], possibleMoves: Set<string>): {
  bestScore1: number, bestScore2: number, bestJointScore: number,
  bestMove: MovesData | null, bestMoves: MovesData | null
} => {
  let bestScore1 = Number.NEGATIVE_INFINITY;
  let bestScore2 = Number.NEGATIVE_INFINITY;
  let bestJointScore = Number.NEGATIVE_INFINITY;
  let bestMove: MovesData | null = null;
  let bestMoves: MovesData | null = null;
  const seen: Set<string> = new Set();

  movesPairs.forEach(movePair => {
    if (!(movePair.move1.sans[0] in seen)) {
      seen.add(movePair.move1.sans[0]);
      const score = calculateScore(state, movePair.move1);
      if (score > 0) {
        possibleMoves.add(movePair.move1.sans[0]);
      }
      if (score > bestScore1) {
        bestMove = movePair.move1;
        bestScore1 = score;
      }
    }

    if ((movePair.move2 === null) || (movePair.moves === null) || !(possibleMoves.has(movePair.move1.sans[0]))) {
      return;
    }

    const score2: number = calculateScore(state, movePair.move2);
    if (score2 < 0) {
      return;
    } else if (score2 > bestScore2) {
      bestScore2 = score2;
    }

    const jointScore: number = calculateScore(state, movePair.moves);
    if (jointScore > bestJointScore) {
      bestJointScore = jointScore;
      bestMoves = movePair.moves;
    }
  });

  return {bestScore1, bestScore2, bestJointScore, bestMove, bestMoves};
};

const getBoxCenters = (boxes: tf.Tensor2D) => {
  const boxCenters: tf.Tensor2D = tf.tidy(() => {
    const l: tf.Tensor2D = tf.slice(boxes, [0, 0], [-1, 1]);
    const r: tf.Tensor2D = tf.slice(boxes, [0, 2], [-1, 1]);
    const b: tf.Tensor2D = tf.slice(boxes, [0, 3], [-1, 1]);
    const cx: tf.Tensor2D = tf.div(tf.add(l, r), 2);
    const cy: tf.Tensor2D = tf.sub(b, tf.div(tf.sub(r, l), 3));
    const boxCenters: tf.Tensor2D = tf.concat([cx, cy], 1);
    return boxCenters;
  });
  return boxCenters;
};

export const getSquares = (boxes: tf.Tensor2D, centers3D: tf.Tensor3D, boundary3D: tf.Tensor3D): number[] => {
  const squares: number[] = tf.tidy(() => {
    const boxCenters3D: tf.Tensor3D = tf.expandDims(getBoxCenters(boxes), 1);
    const dist: tf.Tensor2D = tf.sum(tf.square(tf.sub(boxCenters3D, centers3D)), 2);
    const squares: any = tf.argMin(dist, 1);

    const shiftedBoundary3D: tf.Tensor3D = tf.concat([
      tf.slice(boundary3D, [0, 1, 0], [1, 3, 2]),
      tf.slice(boundary3D, [0, 0, 0], [1, 1, 2]),
    ], 1);

    const nBoxes: number = boxCenters3D.shape[0];

    const a: tf.Tensor2D = tf.squeeze(tf.sub(
      tf.slice(boundary3D, [0, 0, 0], [1, 4, 1]),
      tf.slice(shiftedBoundary3D, [0, 0, 0], [1, 4, 1])
    ), [2]);
    const b: tf.Tensor2D = tf.squeeze(tf.sub(
      tf.slice(boundary3D, [0, 0, 1], [1, 4, 1]),
      tf.slice(shiftedBoundary3D, [0, 0, 1], [1, 4, 1])
    ), [2]);
    const c: tf.Tensor2D = tf.squeeze(tf.sub(
      tf.slice(boxCenters3D, [0, 0, 0], [nBoxes, 1, 1]),
      tf.slice(shiftedBoundary3D, [0, 0, 0], [1, 4, 1])
    ), [2]);
    const d: tf.Tensor2D = tf.squeeze(tf.sub(
      tf.slice(boxCenters3D, [0, 0, 1], [nBoxes, 1, 1]),
      tf.slice(shiftedBoundary3D, [0, 0, 1], [1, 4, 1])
    ), [2]);

    const det: tf.Tensor2D = tf.sub(tf.mul(a, d), tf.mul(b, c));
    const newSquares: tf.Tensor1D = tf.where(
      tf.any(tf.less(det, 0), 1),
      tf.scalar(-1),
      squares
    );

    return newSquares.arraySync();
  });

  return squares;
};

export const getUpdate = (scoresTensor: tf.Tensor2D, squares: number[]) => {
  const update: number[][] = zeros(64, 12);
  const scores: number[][] = scoresTensor.arraySync();

  for (let i = 0; i < squares.length; i++) {
    const square = squares[i];
    if (square == -1) {
      continue;
    }
    for (let j = 0; j < 12; j++) {
      update[square][j] = Math.max(update[square][j], scores[i][j]);
    }
  }
  return update;
};

const updateState = (state: number[][], update: number[][], decay: number=0.5) => {
  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 12; j++) {
      state[i][j] = decay * state[i][j] + (1 - decay) * update[i][j];
    }
  }
  return state;
};

const sanToLan = (board: Chess, san: string): string => {
  board.move(san);
  const history: any = board.history({ verbose: true });
  const lan: string = history[history.length - 1].lan;
  board.undo();
  return lan;
};

export const detect = async (modelRef: any, videoRef: any, keypoints: number[][]):
  Promise<{boxes: tf.Tensor2D, scores: tf.Tensor2D}> => {
  const {image4D, width, height, padding, roi} = getInput(videoRef, keypoints);
  const videoWidth: number = videoRef.current.videoWidth;
  const videoHeight: number = videoRef.current.videoHeight;
  const preds: tf.Tensor3D = modelRef.current.predict(image4D);
  const {boxes, scores} = getBoxesAndScores(preds, width, height, videoWidth, videoHeight, padding, roi);

  tf.dispose([image4D, preds]);

  return {boxes, scores};
};

const getXY = (corner: any, height: number, width: number): number[] => {
  const sx: number = MODEL_WIDTH / width;
  const sy: number = MODEL_HEIGHT / height;
  const XY: number[] = [sx * corner[0], sy * (corner[1] + height + MARKER_DIAMETER)];
  return XY;
};

export const getKeypoints = (cornersRef: any, canvasRef: any): number[][] => {
  const keypoints = CORNER_KEYS.map(x =>
    getXY(cornersRef.current[x], canvasRef.current.height, canvasRef.current.width)
  );
  return keypoints;
};

export const findMultiplePieces = (modelRef: any, videoRef: any, canvasRef: any,
playingRef: any, setText: any, dispatch: any, cornersRef: any, boardRef: any,
movesPairsRef: any, lastMoveRef: any, moveTextRef: any, mode: Mode) => {
  let centersList: number[][][] | null = null;
  let boundaryList: number[][][];
  let centers3DList: tf.Tensor3D[];
  let boundary3DList: tf.Tensor3D[];
  let stateList: number[][][];
  let keypointsList: number[][][];
  let possibleMoves: Set<string>;
  let requestId: number;
  let greedyMoveToTime: { [move: string] : number};

  const loop = async () => {
    if (playingRef.current === false) {
      centersList = null;
    } else {
      if (centersList === null) {
        keypointsList = [getKeypoints(cornersRef, canvasRef), getKeypoints(cornersRef, canvasRef)];
        centers3DList = [];
        boundary3DList = [];
        centersList = [];
        boundaryList = [];
        stateList = [];
        keypointsList.forEach(keypoints => {
          const invTransform = getInvTransform(keypoints);
          const [centers, centers3D] = transformCenters(invTransform);
          const [boundary, boundary3D] = transformBoundary(invTransform);
          centersList.push(centers);
          centers3DList.push(centers3D);
          boundaryList.push(boundary);
          boundary3DList.push(boundary3D);
          stateList.push(zeros(64, 12));
        });
        possibleMoves = new Set<string>;
        greedyMoveToTime = {};
      }
      const startTime: number = performance.now();
      const startTensors: number = tf.memory().numTensors;

      const {boxes, scores} = await detect(modelRef, videoRef, keypointsList[0]);
      const squares: number[] = getSquares(boxes, centers3DList[0], boundary3DList[0]);
      const update: number[][] = getUpdate(scores, squares);
      stateList[0] = updateState(stateList[0], update);
      const {bestScore1, bestScore2, bestJointScore, bestMove, bestMoves} = processState(stateList[0], movesPairsRef.current, possibleMoves);

      const endTime: number = performance.now();
      const fps: string = (1000 / (endTime - startTime)).toFixed(1);

      let hasMove: boolean = false;
      if ((bestMoves !== null) && (mode !== "play")) {
        const move: string = bestMoves.sans[0];
        hasMove = (bestScore2 > 0) && (bestJointScore > 0) && (possibleMoves.has(move));
        if (hasMove) {
          boardRef.current.move(move);
          possibleMoves.clear();
          greedyMoveToTime = {};
        }
      }

      let hasGreedyMove: boolean = false;
      if (bestMove !== null && !(hasMove) && (bestScore1 > 0)) {
        const move: string = bestMove.sans[0];
        if (!(move in greedyMoveToTime)) {
          greedyMoveToTime[move] = endTime;
        }

        const secondElapsed = (endTime - greedyMoveToTime[move]) > 1000;
        const newMove = sanToLan(boardRef.current, move) !== lastMoveRef.current;
        hasGreedyMove = secondElapsed && newMove;
        if (hasGreedyMove) {
          boardRef.current.move(move);
          greedyMoveToTime = {greedyMove: greedyMoveToTime[move]};
        }
      }

      if (hasMove || hasGreedyMove) {
        const greedy = (mode === "play") ? false : hasGreedyMove;
        const payload = makeUpdatePayload(boardRef.current, greedy);
        console.log("payload", payload);
        dispatch(gameUpdate(payload));
      }
      setText([`FPS: ${fps}`, moveTextRef.current]);

      if (centersList) {
        renderMultipleState(canvasRef.current, centersList, boundaryList, stateList);
      }

      tf.dispose([boxes, scores]);

      const endTensors: number = tf.memory().numTensors;
      if (startTensors < endTensors) {
        console.error(`Memory Leak! (${endTensors} > ${startTensors})`);
      }
    }
    requestId = requestAnimationFrame(loop);
  };
  requestId = requestAnimationFrame(loop);

  return () => {
    tf.disposeVariables();
    if (requestId) {
      window.cancelAnimationFrame(requestId);
    }
  };
};