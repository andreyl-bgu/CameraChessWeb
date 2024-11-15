import * as tf from "@tensorflow/tfjs-core";
import { renderMultipleCorners } from "./render/renderMultipleCorners";
import Delaunator from 'delaunator';
import { getPerspectiveTransform, perspectiveTransform } from "./warp";
import { getBoxesAndScores, getInput, getCenters, getMarkerXY, invalidVideo } from "./detect";
import { cornersSet } from '../slices/cornersSlice';
import { CornersDict, CornersPayload, CornersKey } from "../types";
import { CORNER_KEYS, MODEL_WIDTH, MODEL_HEIGHT } from "./constants";
import { clamp } from "./math";
import { NDArray } from "vectorious";

const x: number[] = Array.from({ length: 7 }, (_, i) => i);
const y: number[] = Array.from({ length: 7 }, (_, i) => i);
const GRID: number[][] = y.map(yy => x.map(xx => [xx, yy])).flat();
const IDEAL_QUAD: number[][] = [[0, 1], [1, 1], [1, 0], [0, 0]];

const processBoxesAndScores = async (boxes: tf.Tensor2D, scores: tf.Tensor2D) => {
  const maxScores: tf.Tensor1D = tf.max(scores, 1);
  const argmaxScores: tf.Tensor1D = tf.argMax(scores, 1);
  const nms: tf.Tensor1D = await tf.image.nonMaxSuppressionAsync(boxes, maxScores, 100, 0.3, 0.1);
  const resTensor: tf.Tensor2D = tf.tidy(() => {
    const centers: tf.Tensor2D = getCenters(tf.gather(boxes, nms, 0));
    const cls: tf.Tensor2D = tf.expandDims(tf.gather(argmaxScores, nms, 0), 1);
    const res: tf.Tensor2D = tf.concat([centers, cls], 1);
    return res;
  });
  const res: number[][] = resTensor.arraySync();

  tf.dispose([nms, resTensor, boxes, scores, argmaxScores, maxScores])
  return res
}

const runPiecesModel = async (videoRef: any, piecesModelRef: any): Promise<number[][]> => {
  const videoWidth: number = videoRef.current.videoWidth;
  const videoHeight: number = videoRef.current.videoHeight;

  const {image4D, width, height, padding, roi} = getInput(videoRef);
  const piecesPreds: tf.Tensor3D = piecesModelRef.current.predict(image4D);
  const boxesAndScores = getBoxesAndScores(piecesPreds, width, height, videoWidth, videoHeight, padding, roi);
  const pieces: number[][] = await processBoxesAndScores(boxesAndScores.boxes, boxesAndScores.scores);

  tf.dispose([piecesPreds, image4D, boxesAndScores]);
  return pieces;
}

const runXcornersModel = async (videoRef: any, xcornersModelRef: any, pieces: number[][]): Promise<number[][]> => {
  const keypoints: number[][] = pieces.map(x => [x[0], x[1]]);
  const videoWidth: number = videoRef.current.videoWidth;
  const videoHeight: number = videoRef.current.videoHeight;

  const {image4D, width, height, padding, roi} = getInput(videoRef, keypoints);
  const xcornersPreds: tf.Tensor3D = xcornersModelRef.current.predict(image4D);
  const boxesAndScores = getBoxesAndScores(xcornersPreds, width, height, videoWidth, videoHeight, padding, roi);
  tf.dispose([xcornersPreds, image4D]);

  let xCorners: number[][] = await processBoxesAndScores(boxesAndScores.boxes, boxesAndScores.scores);
  xCorners = xCorners.map(x => [x[0], x[1]]);
  return xCorners;
}

const getQuads = (xCorners: number[][]) => {
  const intXcorners = xCorners.flat().map(x => Math.round(x));
  const delaunay = new Delaunator(intXcorners);
  const triangles = delaunay.triangles;
  const quads = [];
  for (let i = 0; i < triangles.length; i += 3) {
    const t1 = triangles[i];
    const t2 = triangles[i + 1];
    const t3 = triangles[i + 2];
    const quad = [t1, t2, t3, -1];

    for (let j = 0; j < triangles.length; j += 3) {
      if (i === j) {
        continue;
      }
      const cond1 = (t1 === triangles[j] && t2 === triangles[j + 1]) || (t1 === triangles[j + 1] && t2 === triangles[j]);
      const cond2 = (t2 === triangles[j] && t3 === triangles[j + 1]) || (t2 === triangles[j + 1] && t3 === triangles[j]);
      const cond3 = (t3 === triangles[j] && t1 === triangles[j + 1]) || (t3 === triangles[j + 1] && t1 === triangles[j]);
      if ((cond1 || cond2 || cond3)) {
        quad[3] = triangles[j + 2];
        break;
      }
    }

    if (quad[3] !== -1) {
      quads.push(quad.map(x => xCorners[x]));
    }
  }
  return quads;
}

const cdist = (a: number[][], b: number[][]) => {
  const dist = Array.from({ length: a.length }, () => Array(b.length).fill(0));
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      const dx = a[i][0] - b[j][0];
      const dy = a[i][1] - b[j][1];
      dist[i][j] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return dist;
}

const calculateOffsetScore = (warpedXcorners: number[][], shift: number[]) => {
  const grid = GRID.map(x => [x[0] + shift[0], x[1] + shift[1]]);
  const dist = cdist(grid, warpedXcorners);

  let assignmentCost = 0;
  for (let i = 0; i < dist.length; i++) {
    assignmentCost += Math.min(...dist[i]);
  }
  const score = 1 / (1 + assignmentCost);

  return score;
}

const findOffset = (warpedXcorners: number[][]) => {
  const bestOffset = [0, 0];
  for (let i = 0; i < 2; i++) {
    let low = -7;
    let high = 1;
    const scores: any = {};
    while ((high - low) > 1) {
      const mid = (high + low) >> 1;
      [mid, mid + 1].forEach(x => {
        if (!(x in scores)) {
          const shift = [0, 0];
          shift[i] = x;
          scores[x] = calculateOffsetScore(warpedXcorners, shift);
        }
      });
      if (scores[mid] > scores[mid + 1]) {
        high = mid
      } else {
        low = mid
      }
    }
    bestOffset[i] = low + 1;
  }

  return bestOffset;
}

const scoreQuad = (quad: number[][], xCorners: number[][]): [number, NDArray, number[]] => {
  const M: NDArray = getPerspectiveTransform(IDEAL_QUAD, quad);
  const warpedXcorners: number[][] = perspectiveTransform(xCorners, M);
  const offset: number[] = findOffset(warpedXcorners);

  const score: number = calculateOffsetScore(warpedXcorners, offset);
  return [score, M, offset]
}

const findCornersFromXcorners = (xCorners: number[][]) => {
  const quads: number[][][] = getQuads(xCorners);
  if (quads.length == 0) {
    return;
  }

  let bestScore: number;
  let bestM: NDArray;
  let bestOffset: number[];
  [bestScore, bestM, bestOffset] = scoreQuad(quads[0], xCorners);
  for (let i = 1; i < quads.length; i++) {
    const [score, M, offset] = scoreQuad(quads[i], xCorners);
    if (score > bestScore) {
      bestScore = score;
      bestM = M;
      bestOffset = offset;
    }
  }

  const invM = bestM.inv()
  const warpedCorners = [[bestOffset[0] - 1, bestOffset[1] - 1],
                         [bestOffset[0] - 1, bestOffset[1] + 7],
                         [bestOffset[0] + 7, bestOffset[1] + 7],
                         [bestOffset[0] + 7, bestOffset[1] - 1]]
  const corners = perspectiveTransform(warpedCorners, invM);

  // Clip bad corners
  for (let i = 0; i < 4; i++) {
    corners[i][0] = clamp(corners[i][0], 0, MODEL_WIDTH);
    corners[i][1] = clamp(corners[i][1], 0, MODEL_HEIGHT);
  }

  return corners;
}

const calculateKeypoints = (blackPieces: number[][], whitePieces: number[][], corners: number[][]) => {
  const blackCenter = getCenter(blackPieces);
  const whiteCenter = getCenter(whitePieces);

  let bestShift = 0;
  let bestScore = 0;
  for (let shift = 0; shift < 4; shift++) {
    const cw = [(corners[shift % 4][0] + corners[(shift + 1) % 4][0]) / 2,
                (corners[shift % 4][1] + corners[(shift + 1) % 4][1]) / 2];
    const cb = [(corners[(shift + 2) % 4][0] + corners[(shift + 3) % 4][0]) / 2,
                (corners[(shift + 2) % 4][1] + corners[(shift + 3) % 4][1]) / 2];
    const score = 1 / (1 + euclidean(whiteCenter, cw) + euclidean(blackCenter, cb));
    if (score > bestScore) {
      bestScore = score;
      bestShift = shift;
    }
  }

  const keypoints: CornersDict = {
    "a1": corners[bestShift % 4],
    "h1": corners[(bestShift + 1) % 4],
    "h8": corners[(bestShift + 2) % 4],
    "a8": corners[(bestShift + 3) % 4]
  }
  return keypoints
}

const getCenter = (points: number[][]) => {
  let center = points.reduce((a, b) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
  center = center.map(x => x / points.length);
  return center
}

const euclidean = (a: number[], b: number[]) => {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dist = Math.sqrt((dx * dx) + (dy * dy))
  return dist;
}

export const _findMultipleCorners = async (piecesModelRef: any, xcornersModelRef: any, videoRef: any,
  canvasRef: any, dispatch: any, setText: any) => {
  if (invalidVideo(videoRef)) {
    return;
  }

  const pieces = await runPiecesModel(videoRef, piecesModelRef);
  const blackPieces = pieces.filter((x: number[]) => (x[2] <= 5));
  const whitePieces = pieces.filter((x: number[]) => (x[2] > 5));
  if ((blackPieces.length == 0) || (whitePieces.length == 0)) {
    setText(["No pieces to label corners"]);
    return;
  }

  const xCorners = await runXcornersModel(videoRef, xcornersModelRef, pieces);
  if (xCorners.length < 5) {
    setText(["Need ≥5 xCorners", `Detected ${xCorners.length}`]);
    return;
  }

  const corners1 = findCornersFromXcorners(xCorners);
  if (corners1 === undefined) {
    setText(["Failed to find corners for board 1"]);
    return;
  }

  const keypoints1: CornersDict = calculateKeypoints(blackPieces, whitePieces, corners1);

  const xCorners2 = await runXcornersModel(videoRef, xcornersModelRef, pieces);
  if (xCorners2.length < 5) {
    setText(["Need ≥5 xCorners", `Detected ${xCorners2.length}`]);
    return;
  }

  const corners2 = findCornersFromXcorners(xCorners2);
  if (corners2 === undefined) {
    setText(["Failed to find corners for board 2"]);
    return;
  }

  const keypoints2: CornersDict = calculateKeypoints(blackPieces, whitePieces, corners2);

  CORNER_KEYS.forEach((key: string) => {
    const xy1: number[] = keypoints1[key];
    const payload1: CornersPayload = {
      "xy": getMarkerXY(xy1, canvasRef.current.height, canvasRef.current.width),
      "key": key as CornersKey
    };
    dispatch(cornersSet(payload1));

    const xy2: number[] = keypoints2[key];
    const payload2: CornersPayload = {
      "xy": getMarkerXY(xy2, canvasRef.current.height, canvasRef.current.width),
      "key": key as CornersKey
    };
    dispatch(cornersSet(payload2));
  });

  renderMultipleCorners(canvasRef.current, xCorners, xCorners2);
  setText(["Found corners for both boards", "Ready to record"]);

  // Dispose of tensors
  tf.dispose([pieces, xCorners, xCorners2]);
};

export const findMultipleCorners = async (piecesModelRef: any, xcornersModelRef: any, videoRef: any, canvasRef: any,
  dispatch: any, setText: any) => {
  const startTensors = tf.memory().numTensors;

  await _findMultipleCorners(piecesModelRef, xcornersModelRef, videoRef, canvasRef, dispatch, setText);

  const endTensors = tf.memory().numTensors;
  if (startTensors < endTensors) {
    console.error(`Memory Leak! (${endTensors} > ${startTensors})`);
  }

  return () => {
    tf.disposeVariables();
  };
};