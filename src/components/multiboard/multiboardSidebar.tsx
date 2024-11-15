import { MultiCornersButton, Sidebar, RecordButton, StopButton, StudyButton, DeviceButton } from "../common";
import { SetBoolean, SetNumber, SetStringArray, SetStudy, Study } from "../../types";
import BoardNumberInput from "./boardNumberInput";

const MultiboardSidebar = ({ piecesModelRef, xcornersModelRef, videoRef, canvasRef, sidebarRef,
                               playing, setPlaying, text, setText, study, setStudy, setBoard1Number, setBoard2Number }: {
    piecesModelRef: any, xcornersModelRef: any, videoRef: any, canvasRef: any,
    sidebarRef: any, playing: boolean, setPlaying: SetBoolean,
    text: string[], setText: SetStringArray,
    study: Study | null, setStudy: SetStudy,
    setBoard1Number: SetNumber, setBoard2Number: SetNumber
}) => {
    const inputStyle = {
        display: playing ? "none" : "inline-block"
    }
    return (
        <Sidebar sidebarRef={sidebarRef} playing={playing} text={text} setText={setText} >
            <li className="my-1" style={inputStyle}>
                <DeviceButton videoRef={videoRef} />
            </li>
            <li className="my-1" style={inputStyle}>
                <StudyButton study={study} setStudy={setStudy} onlyBroadcasts={true} />
            </li>
            <li className="my-1" style={inputStyle}>
                <BoardNumberInput setBoardNumber={setBoard1Number} label="Board 1" />
            </li>
            <li className="my-1" style={inputStyle}>
                <BoardNumberInput setBoardNumber={setBoard2Number} label="Board 2" />
            </li>
            <li className="my-1" style={inputStyle}>
                <MultiCornersButton piecesModelRef={piecesModelRef} xcornersModelRef={xcornersModelRef}
                                    videoRef={videoRef} canvasRef={canvasRef} setText={setText} />
            </li>
            <li className="my-1">
                <div className="btn-group w-100" role="group">
                    <RecordButton playing={playing} setPlaying={setPlaying} />
                    <StopButton setPlaying={setPlaying} setText={setText} />
                </div>
            </li>
        </Sidebar>
    );
};

export default MultiboardSidebar;