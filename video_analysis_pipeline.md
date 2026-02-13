# Video Analysis Pipeline

End-to-end workflow: upload/stage video -> per-frame CV analysis -> optional scene understanding -> optional corpus build/ingest -> signed results.

```mermaid
flowchart TD
    %% Styling
    classDef api fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef decision fill:#fff8e1,stroke:#e65100,stroke-width:2px;
    classDef storage fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    %% Stage 1: Upload + staging
    User([User / Client]):::api --> AnalyzeEndpoint[POST /analyze-video]:::api
    AnalyzeEndpoint --> ReturnJob[[202 + job_id]]:::output
    AnalyzeEndpoint --> LocalStage[Local staged source\nTEMP_MEDIA_DIR/job_id/input/source.ext]:::process
    LocalStage --> ProcessVideo[Background task:\nprocess_video]:::process
    ProcessVideo --> UploadSource[Upload source video]:::process
    UploadSource --> R2Source[(R2: jobs/job_id/input/source.ext)]:::storage
    R2Source --> VerifySource{Source upload\nverified?}:::decision

    %% Stage 2: Scene detection + keyframes
    VerifySource -->|yes| SceneDetect[scene.detect_scenes]:::process
    SceneDetect --> KeyframeExtract[scene.extract_keyframes]:::process
    KeyframeExtract --> SaveOriginal[scene.save_original_frames]:::process
    SaveOriginal --> R2Original[(R2: frames/original)]:::storage

    %% Stage 3: Per-frame CV analysis
    KeyframeExtract --> AnalyzeFrame[analysis.analyze_frame\n(per keyframe)]:::process
    AnalyzeFrame --> YOLOSeg[YOLO segmentation]:::process
    AnalyzeFrame --> YOLODet[YOLO detection]:::process
    AnalyzeFrame --> MTCNN[MTCNN face detection]:::process
    YOLODet --> ObjTrack[ObjectTrackTracker]:::process
    MTCNN --> FaceTrack[FaceIdentityTracker]:::process
    AnalyzeFrame -. optional .-> Enrichment[Optional enrichers:\nOCR, pose, action, camera motion, quality]:::process

    YOLOSeg --> PersistViz[Persist visualization frames]:::process
    YOLODet --> PersistViz
    MTCNN --> PersistViz
    PersistViz --> R2Seg[(R2: frames/seg)]:::storage
    PersistViz --> R2Det[(R2: frames/det)]:::storage
    PersistViz --> R2Face[(R2: frames/face)]:::storage

    AnalyzeFrame --> PersistFrameArtifacts[Persist frame artifacts]:::process
    ObjTrack --> PersistFrameArtifacts
    FaceTrack --> PersistFrameArtifacts
    Enrichment --> PersistFrameArtifacts
    PersistFrameArtifacts --> R2FrameJSON[(R2: analysis/json/frame_N.json)]:::storage
    PersistFrameArtifacts --> R2FrameTOON[(R2: analysis/toon/frame_N.toon)]:::storage
    PersistFrameArtifacts --> FrameResults[frame_results[]]:::output

    %% Stage 4: Optional scene understanding pipeline
    SceneDetect --> SceneGate{ENABLE_SCENE_UNDERSTANDING_PIPELINE}:::decision
    FrameResults --> SceneGate
    SceneGate -->|true| BuildScenePackets[Build scene packets (TOON)]:::process
    BuildScenePackets --> R2ScenePackets[(R2: scene/packets/scene_N.toon)]:::storage
    BuildScenePackets --> GenerateNarratives[Generate scene narratives]:::process
    GenerateNarratives --> R2SceneNarratives[(R2: scene/narratives/scene_N.json)]:::storage
    GenerateNarratives --> BuildSynopsis[Refine video synopsis]:::process
    BuildSynopsis --> R2Synopsis[(R2: summary/synopsis.json)]:::storage
    BuildSynopsis --> SceneOutputs[scene_narratives + video_synopsis]:::output
    SceneGate -->|false| SceneOutputs

    %% Stage 5: Optional corpus build + ingest
    FrameResults --> CorpusGate{ENABLE_CORPUS_PIPELINE}:::decision
    SceneOutputs --> CorpusGate
    CorpusGate -->|true| BuildCorpus[corpus.build:\ngraph + retrieval + embeddings]:::process
    BuildCorpus --> R2Graph[(R2: corpus/graph/bundle.json)]:::storage
    BuildCorpus --> R2Retrieval[(R2: corpus/rag/bundle.json)]:::storage
    BuildCorpus --> R2Embeddings[(R2: corpus/embeddings/bundle.json)]:::storage
    BuildCorpus --> IngestGate{ENABLE_CORPUS_INGEST}:::decision
    IngestGate -->|true| IngestCorpus[corpus_ingest.ingest_corpus]:::process
    IngestCorpus --> Neo4j[(Neo4j)]:::storage
    IngestCorpus --> Pgvector[(pgvector)]:::storage

    %% Stage 6: Completion + result serving
    FrameResults --> AssemblePayload[Assemble job payload]:::process
    SceneOutputs --> AssemblePayload
    BuildCorpus --> AssemblePayload
    CorpusGate -->|false| AssemblePayload
    IngestGate -->|false| AssemblePayload
    AssemblePayload --> VerifyArtifacts[Verify required R2 artifacts]:::process
    VerifyArtifacts --> CompleteJob[jobs.complete_job]:::output
    CompleteJob --> ResultsEndpoint[GET /results/{job_id}]:::api
    ResultsEndpoint --> SignedResult[[Signed URLs + result payload]]:::output
    CompleteJob --> FinalizeSource[Finalize local source\n(delete or retain)]:::process
    VerifySource -->|no (fail job)| FinalizeSource
```
