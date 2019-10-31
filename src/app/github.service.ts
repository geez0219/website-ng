import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class GithubService {

  examples_url: string = 'https://api.github.com/repos/fastestimator/fastestimator/contents/apphub?Accept=application/vnd.github.v3+json';

  constructor(private http: HttpClient) { }

  getExampleList() {
    return this.http.get(this.examples_url);
  }
}
