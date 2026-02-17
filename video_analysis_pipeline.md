# Video Analysis Pipeline

End-to-end workflow: upload/stage video -> per-frame CV analysis -> video-level tracking -> signed results.

The runtime pipeline now stops after CV and tracking outputs. There is no scene-understanding queue stage and no corpus build stage in `process_video()`.

```mermaid
flowchart TD
    classDef api fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff8e1,stroke:#e65100,stroke-width:2px
    classDef storage fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px

    subgraph S1["1 · Upload & Staging"]
        User([User / Client]):::api
        AnalyzeEndpoint[POST /analyze-video]:::api
        ReturnJob[[202 + job_id]]:::output
        LocalStage[Stage source locally]:::process
        ProcessVideo[Background: process_video]:::process
        UploadSource[Upload source to R2]:::process
        R2Source[(R2: input/source)]:::storage
        VerifySource{Source upload verified?}:::decision

        User --> AnalyzeEndpoint
        AnalyzeEndpoint --> ReturnJob
        AnalyzeEndpoint --> LocalStage
        LocalStage --> ProcessVideo
        ProcessVideo --> UploadSource
        UploadSource --> R2Source
        R2Source --> VerifySource
    end

    subgraph S2["2 · Scene Detection & Keyframes"]
        SceneDetect[Detect scenes]:::process
        KeyframeExtract[Extract midpoint keyframe per scene]:::process
        SaveOriginal[Save original frames]:::process
        R2Original[(R2: frames/original)]:::storage

        SceneDetect --> KeyframeExtract
        KeyframeExtract --> SaveOriginal
        SaveOriginal --> R2Original
    end

    VerifySource -->|yes| SceneDetect

    subgraph S3["3 · Per-frame CV Analysis"]
        AnalyzeFrame[analyze_frame per keyframe]:::process
        YOLOSeg[YOLO segmentation]:::process
        YOLODet[YOLO detection]:::process
        MTCNN[MTCNN face detection]:::process
        Enrichment[Optional enrichment hooks]:::process

        PersistViz[Persist visualization frames]:::process
        R2Seg[(R2: frames/seg)]:::storage
        R2Det[(R2: frames/det)]:::storage
        R2Face[(R2: frames/face)]:::storage

        PersistJSON[Persist frame JSON artifact]:::process
        R2FrameJSON[(R2: analysis/json/frame_N.json)]:::storage
        FrameResults[frame_results]:::output

        AnalyzeFrame --> YOLOSeg
        AnalyzeFrame --> YOLODet
        AnalyzeFrame --> MTCNN
        AnalyzeFrame -. optional .-> Enrichment

        YOLOSeg --> PersistViz
        YOLODet --> PersistViz
        MTCNN --> PersistViz
        PersistViz --> R2Seg
        PersistViz --> R2Det
        PersistViz --> R2Face

        AnalyzeFrame --> PersistJSON
        PersistJSON --> R2FrameJSON
        PersistJSON --> FrameResults
    end

    KeyframeExtract --> AnalyzeFrame

    subgraph S4["4 · Video-level Tracking"]
        ObjSummary[Object tracking summary]:::process
        ObjResult[video_object_tracks]:::output

        FaceGate{ENABLE_FACE_IDENTITY_PIPELINE}:::decision
        TrackingFrames[Sample tracking frames]:::process
        FaceIdentity[Face identity clustering]:::process
        PersonFusion[Person fusion]:::process
        FaceResult[video_face_identities]:::output
        PersonResult[video_person_tracks]:::output

        ObjSummary --> ObjResult
        FaceGate -->|enabled| TrackingFrames
        TrackingFrames --> FaceIdentity
        FaceIdentity --> FaceResult
        FaceIdentity --> PersonFusion
        PersonFusion --> PersonResult
        FaceGate -->|disabled| PersonResult
    end

    FrameResults --> ObjSummary
    FrameResults --> FaceGate
    SceneDetect --> TrackingFrames

    subgraph S5["5 · Completion & Results"]
        AssemblePayload[Assemble payload]:::process
        VerifyArtifacts[Verify required R2 artifacts]:::process
        CompleteJob[complete_job]:::output
        ResultsEndpoint[GET /results/{job_id}]:::api
        SignedResult[[Signed frame URLs + analysis payload]]:::output
        FinalizeSource[Finalize local source cleanup]:::process

        AssemblePayload --> VerifyArtifacts
        VerifyArtifacts --> CompleteJob
        CompleteJob --> ResultsEndpoint
        ResultsEndpoint --> SignedResult
        CompleteJob --> FinalizeSource
    end

    FrameResults --> AssemblePayload
    ObjResult --> AssemblePayload
    FaceResult --> AssemblePayload
    PersonResult --> AssemblePayload
    VerifySource -->|fail| FinalizeSource
```
