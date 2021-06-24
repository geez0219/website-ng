import { Injectable, Output, EventEmitter } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { Version } from './version';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class GlobalService {
  versions: Version[];
  selectedVersion: string;
  latestVersion: string;

  version = new Subject();

  loading: boolean;
  @Output() change: EventEmitter<boolean> = new EventEmitter();

  constructor(private http: HttpClient) { }

  setLoading() {
    this.loading = true;
    this.change.emit(this.loading);
  }

  resetLoading() {
    this.loading = false;
    this.change.emit(this.loading);
  }

  isLatest(element: Version, index: number, array: Version[]) {
    return (element.latest === true);
  }

  setVersions(version: Version[]) {
    this.versions = version;
    this.latestVersion = this.versions.filter(this.isLatest)[0].name;
    this.selectedVersion = this.latestVersion;
  }

  getLatestVersion() {
    return this.latestVersion;
  }

  getSelectedVersion() {
    return this.selectedVersion;
  }

  setCurrentVersion(selectedVersion: string) {
    this.selectedVersion = selectedVersion;
    this.versions.filter(this.isLatest)[0].latest = false;
    this.versions.find(version => version.name === selectedVersion).latest = true;
  }

  getVersions(){
    return this.versions;
  }

  setSelectedVersion(selectedVersion: Version) {
    this.selectedVersion = selectedVersion.name;
  }
}
