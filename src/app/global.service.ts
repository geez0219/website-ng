import { Injectable, Output, EventEmitter } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { Branch } from './branch';

@Injectable({
  providedIn: 'root'
})
export class GlobalService {

  examples_url: string = 'https://api.github.com/repos/fastestimator/fastestimator/contents/apphub?Accept=application/vnd.github.v3+json';

  branches: Branch[];
  selectedBranch: string;

  loading: boolean;
  @Output() change: EventEmitter<boolean> = new EventEmitter();

  constructor(private http: HttpClient) { }

  getExampleList() {
    return this.http.get(this.examples_url);
  }

  toggleLoading() {
    this.loading = !this.loading;
    this.change.emit(this.loading);
  }

  setLoading() {
    this.loading = true;
    this.change.emit(this.loading);
  }

  resetLoading() {
    this.loading = false;
    this.change.emit(this.loading);
  }

  isLatest(element: Branch, index: number, array: Branch[]) {
    return (element.latest == true);
  }

  setBranches(branches: Branch[]) {
    this.branches = branches;
    this.selectedBranch = this.branches.filter(this.isLatest)[0].name;
  }

  getSelectedBranch() {
    return this.selectedBranch;
  }

  setCurrentBranch(selectedBranch: string) {
    this.selectedBranch = selectedBranch;
  }
}
