# CHANGELOG

<!-- version list -->

## v0.3.0 (2026-09-07)

### Chores

- **lock**: Sync the release version
  ([`cacc6e1`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/cacc6e1f06158c066337a0f3ed7c988f61abc313))

### Features

- Support incremental row snapshots
  ([`ed3a677`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/ed3a677bbd7578a13f4a4e1813cecf72da71d89f))

### Performance Improvements

- **server**: Make snapshot cache TTL configurable
  ([`500203f`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/500203f84b2e5e04ae1f70d5fcc81214c8bb8632))


## v0.2.0 (2026-09-06)

### Chores

- Attribute release commits to the Janitor identity
  ([`2b095d2`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/2b095d2aaf25eaafdd3140978bad6f32e55f928b))

- Attribute the copyright to SQUAD Lab
  ([`22a7d6c`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/22a7d6c899693bda2014012258ba5d8b16fd571b))

- **deps**: Require QCoDeS 0.59 for examples
  ([`23a20e6`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/23a20e631c0bfd328265074a391c79585138c485))

### Continuous Integration

- Detect a no-op release by comparing tags
  ([`fdfd322`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/fdfd32273a61f46bce54e5d7637c60e1e4d69884))

### Documentation

- Add a PyPI version badge
  ([`21fb710`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/21fb710468a43720fd3b540445cdb41817ecbe2d))

- Fix status badges going blank after a release
  ([`142bfcc`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/142bfcc88a5d58fac1c769937fd6708ab89b3f4b))

- Update installation instructions to reference the preview branch
  ([`1acf7a0`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/1acf7a05ea107acb2075123e33861dab0644b07b))

### Features

- Align live publishing with Qanary
  ([`c45c70e`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/c45c70efa5119fc60c3a33c2d3f3cd39b095d4dc))

### Testing

- Pin the polling cache to one build per cache lifetime
  ([`452cf21`](https://gitlab.com/squad-lab/qimchi-connect/-/commit/452cf21be651a2219aaa489e53a241b3b94d23f9))

### Breaking Changes

- QCUtilsSnapshotProvider is replaced by QanarySnapshotProvider, and the live registry moves from
  ~/.qcutils to QIMCHI_HOME or ~/.qimchi.


## v0.1.0 (2026-09-05)

- Initial Release
