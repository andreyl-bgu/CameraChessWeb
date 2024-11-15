import { LABELS, PALETTE } from "../constants";
import { setupCtx, drawPoints, drawPolygon, drawBoxes } from "./common";

export const renderMultipleState = (canvasRef: any, centersList: number[][][], boundaryList: number[][][], stateList: number[][][]) => {
  const [ctx, fontHeight, lineWidth, sx, sy] = setupCtx(canvasRef);

  centersList.forEach(centers => {
    drawPoints(ctx, centers, "blue", sx, sy);
  });

  boundaryList.forEach(boundary => {
    drawPolygon(ctx, boundary, "blue", sx, sy);
  });

  stateList.forEach((state, index) => {
    const boxes = state.map((cell, i) => {
      let bestScore = 0.1;
      let bestPiece = -1;
      for (let j = 0; j < 12; j++) {
        if (cell[j] > bestScore) {
          bestScore = cell[j];
          bestPiece = j;
        }
      }

      if (bestPiece === -1) {
        return null;
      }

      const color = PALETTE[bestPiece % PALETTE.length];
      const text = `${LABELS[bestPiece]}:${Math.round(100 * bestScore)}`;
      return { colour: color, cx: centersList[index][i][0] * sx, cy: centersList[index][i][1] * sy, text };
    }).filter(box => box !== null);

    drawBoxes(ctx, boxes, fontHeight, lineWidth);
  });
};