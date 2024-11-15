import { findMultipleCorners } from "../../utils/findMultipleCorners";
import { useDispatch } from 'react-redux';
import SidebarButton from "./sidebarButton";

const MultiCornersButton = ({ piecesModelRef, xcornersModelRef, videoRef, canvasRef, setText }:
  {piecesModelRef: any, xcornersModelRef: any, videoRef: any, canvasRef: any, setText: any}) => {
  const dispatch = useDispatch();

  const handleClick = (e: any) => {
    e.preventDefault();

    // Call findMultipleCorners instead of findCorners
    findMultipleCorners(piecesModelRef, xcornersModelRef, videoRef, canvasRef, dispatch, setText);
  }

  return (
    <SidebarButton onClick={handleClick}>
      Find Corners for Both Boards
    </SidebarButton>
  );
};

export default MultiCornersButton;