import { drawPoints, setupCtx } from "./common";

export const renderMultipleCorners = (canvasRef: any, xCorners1: number[][], xCorners2: number[][]) => {
  const [ctx, _, __, sx, sy] = setupCtx(canvasRef);

  // Render corners for the first board
  drawPoints(ctx, xCorners1, "blue", sx, sy);

  // Render corners for the second board
  drawPoints(ctx, xCorners2, "red", sx, sy);
}