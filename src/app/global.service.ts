import { Injectable, Output, EventEmitter } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class GlobalService {

  examples_url: string = 'https://api.github.com/repos/fastestimator/fastestimator/contents/apphub?Accept=application/vnd.github.v3+json';
  
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
}
