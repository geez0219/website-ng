import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['./tutorial.component.css']
})
export class TutorialComponent implements OnInit {
  apiList: string[];
  currentSelection: string;
  currentAPI: string;

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.apiList = ['network', 'pipeline'];
    this.currentSelection = 'assets/api/network.md';
    
    this.getAPIText();

    console.log(this.currentAPI);
  }

  updateCurrentAPI(api: string) {
    this.currentSelection = 'assets/api/' + api + '.md';

    this.getAPIText();
  }

  getAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPI = data;
    });
  }

}
